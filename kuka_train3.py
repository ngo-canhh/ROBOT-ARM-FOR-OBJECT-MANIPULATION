# --- IMPROVED KUKA TRAIN SCRIPT WITH MEMORY MANAGEMENT AND COMMAND LINE ARGUMENTS ---

import os
import os.path as path
import glob
import time
import numpy as np
import torch
import functools
import re
import argparse  # Add this import for command line arguments

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Import our environment and custom extractor
try:
    from kuka_vision_grasping_env4 import KukaVisionGraspingEnv
    from custom_extractor3 import CustomExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure kuka_vision_grasping_env2.py and custom_extractor.py are in the current directory or PYTHONPATH.")
    exit(1)

# --- Command Line Arguments Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Kuka robot agent using SAC algorithm with customizable parameters.")
    
    # Environment parameters
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    
    # Curriculum Learning parameters
    parser.add_argument('--curriculum', action='store_true', default=True, help='Enable curriculum learning')
    parser.add_argument('--initial_ws_size', type=float, default=0.05, help='Initial workspace size (side length in meters) for curriculum learning')
    parser.add_argument('--final_ws_size', type=float, default=0.3, help='Final workspace size (side length in meters) for curriculum learning')
    parser.add_argument('--success_threshold', type=float, default=0.6, help='Success rate threshold to advance curriculum')
    parser.add_argument('--eval_window', type=int, default=100, help='Number of episodes to compute success rate')
    parser.add_argument('--curriculum_steps', type=int, default=5, help='Number of curriculum steps from initial to final workspace size')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps for training')
    parser.add_argument('--buffer_size', type=int, default=175_000, help='Size of the replay buffer')
    parser.add_argument('--learning_starts', type=int, default=30_000, help='Number of steps before starting to learn')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for each training update')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--policy_features_dim', type=int, default=512, help='Feature dimension for policy network')
    
    # Checkpoint and logging parameters
    parser.add_argument('--checkpoint_freq', type=int, default=40_000, help='Frequency of saving checkpoints (will be multiplied by n_envs)')
    parser.add_argument('--reward_log_freq', type=int, default=5000, help='Frequency of logging rewards')
    parser.add_argument('--eval_freq', type=int, default=40_000, help='Frequency of evaluation (will be multiplied by n_envs)')
    parser.add_argument('--log_interval', type=int, default=50, help='SB3 default logging frequency')
    
    # Directories
    parser.add_argument('--log_dir_base', type=str, default=os.path.join(os.getcwd(), "logs_kuka_sac"), 
                        help='Base directory for logs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, 
                        help='Computing device to use (cuda, cpu, or auto for automatic detection)')
    
    return parser.parse_args()

# --- Configuration ---
def setup_config(args):
    config = {
        'N_ENVS': args.n_envs,
        'LOG_DIR_BASE': args.log_dir_base,
        'CHECKPOINT_DIR': os.path.join(args.log_dir_base, "checkpoints"),
        'TENSORBOARD_LOG_DIR': os.path.join(args.log_dir_base, "tensorboard"),
        'FINAL_MODEL_PATH': os.path.join(args.log_dir_base, "sac_kuka_final_model.zip"),
        
        'TOTAL_TIMESTEPS': args.total_timesteps,
        'CHECKPOINT_FREQ': args.checkpoint_freq * args.n_envs,
        'REWARD_LOG_FREQ': args.reward_log_freq,
        'EVAL_FREQ': args.eval_freq * args.n_envs,
        'BUFFER_SIZE': args.buffer_size,
        'LEARNING_STARTS': args.learning_starts,
        'BATCH_SIZE': args.batch_size,
        'POLICY_FEATURES_DIM': args.policy_features_dim,
        'LOG_INTERVAL': args.log_interval,
        'LEARNING_RATE': args.learning_rate,
        
        # Curriculum Learning
        'CURRICULUM_ENABLED': args.curriculum,
        'INITIAL_WS_SIZE': args.initial_ws_size,
        'FINAL_WS_SIZE': args.final_ws_size,
        'SUCCESS_THRESHOLD': args.success_threshold,
        'EVAL_WINDOW': args.eval_window,
        'CURRICULUM_STEPS': args.curriculum_steps,
        'CURRENT_CURRICULUM_STAGE': 0,  # Starting at stage 0
        'CURRENT_WS_SIZE': args.initial_ws_size,  # Start with initial size
        
        'BASE_SEED': args.seed
    }
    
    # Determine computing device
    if args.device == 'cuda' and torch.cuda.is_available():
        config['DEVICE'] = 'cuda'
    elif args.device == 'cpu':
        config['DEVICE'] = 'cpu'
    else:  # Auto or invalid value
        config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create directories if they don't exist
    os.makedirs(config['LOG_DIR_BASE'], exist_ok=True)
    os.makedirs(config['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(config['TENSORBOARD_LOG_DIR'], exist_ok=True)
    
    return config

# --- Set seeds for reproducibility ---
def set_seeds(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = False if torch.cuda.is_available() else True
    np.random.seed(seed)

# --- Environment creation function ---
def make_env(rank, seed=0, max_steps=300, log_dir=None):
    """
    Utility function to create environment for VecEnv.
    This function is called inside each worker process.
    """
    try:
        # Create and monitor environment
        env = Monitor(
            KukaVisionGraspingEnv(render_mode='rgb_array', max_steps=max_steps),
            os.path.join(log_dir, f'env_{rank}') if log_dir else None
        )
        # Reset with unique seed
        env.reset(seed=seed + rank)
        return env
    except Exception as e:
        print(f"!!! Error in worker {rank} creating env: {e}")
        import traceback
        traceback.print_exc()
        raise e

# --- Functions for checkpoint management ---
def get_step_from_checkpoint_filename(filename):
    """Extract step number from a checkpoint filename."""
    try:
        match = re.search(r'_(\d+)_steps', filename)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0

def get_sorted_checkpoints(checkpoint_dir):
    """Get a list of checkpoint filenames sorted by step number."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "kuka_pushing_model_*_steps.zip"))
    return sorted(checkpoint_files, key=get_step_from_checkpoint_filename)

def get_replay_buffer_filename(model_filename):
    """Get the corresponding replay buffer filename from a model filename."""
    dir_name = os.path.dirname(model_filename)
    base_name = os.path.basename(model_filename)
    # Replace the .zip extension with _replay_buffer.pkl
    buffer_name = base_name.replace('.zip', '_replay_buffer.pkl')
    return os.path.join(dir_name, buffer_name)

def get_vecnormalize_filename(model_filename):
    """Get the corresponding VecNormalize filename from a model filename."""
    dir_name = os.path.dirname(model_filename)
    base_name = os.path.basename(model_filename)
    # Replace the .zip extension with _vecnormalize.pkl
    vecnorm_name = base_name.replace('.zip', '_vecnormalize.pkl')
    return os.path.join(dir_name, vecnorm_name)

def get_curriculum_filename(model_filename):
    """Get the corresponding curriculum state filename from a model filename."""
    dir_name = os.path.dirname(model_filename)
    base_name = os.path.basename(model_filename)
    # Replace the .zip extension with _curriculum.npy
    curriculum_name = base_name.replace('.zip', '_curriculum.npy')
    return os.path.join(dir_name, curriculum_name)

def cleanup_old_checkpoints(checkpoint_dir):
    """
    Clean up old checkpoints before saving a new one:
    1. Delete all replay buffers (to free space for the new one)
    2. Delete all VecNormalize files
    3. Delete all curriculum state files
    4. Keep only the latest model checkpoint, delete older ones
    """
    # Get all checkpoints sorted by step number
    sorted_checkpoints = get_sorted_checkpoints(checkpoint_dir)
    
    if not sorted_checkpoints:
        print("No existing checkpoints found to clean up.")
        return
    
    # Delete all replay buffers, VecNormalize files, and curriculum state files (we'll save new ones)
    print("Deleting all existing replay buffers, VecNormalize files, and curriculum state files...")
    for checkpoint in sorted_checkpoints:
        # Remove replay buffer
        buffer_file = get_replay_buffer_filename(checkpoint)
        if os.path.exists(buffer_file):
            print(f"Removing replay buffer: {buffer_file}")
            try:
                os.remove(buffer_file)
            except OSError as e:
                print(f"Error removing replay buffer {buffer_file}: {e}")
        
        # Remove VecNormalize file
        vecnorm_file = get_vecnormalize_filename(checkpoint)
        if os.path.exists(vecnorm_file):
            print(f"Removing VecNormalize file: {vecnorm_file}")
            try:
                os.remove(vecnorm_file)
            except OSError as e:
                print(f"Error removing VecNormalize file {vecnorm_file}: {e}")
        
        # Remove curriculum state file
        curriculum_file = get_curriculum_filename(checkpoint)
        if os.path.exists(curriculum_file):
            print(f"Removing curriculum state file: {curriculum_file}")
            try:
                os.remove(curriculum_file)
            except OSError as e:
                print(f"Error removing curriculum state file {curriculum_file}: {e}")
    
    # Keep only the latest model checkpoint (if there are multiple)
    if len(sorted_checkpoints) > 1:
        latest_checkpoint = sorted_checkpoints[-1]
        for old_checkpoint in sorted_checkpoints[:-1]:
            print(f"Removing old model checkpoint: {old_checkpoint}")
            try:
                os.remove(old_checkpoint)
            except OSError as e:
                print(f"Error removing old checkpoint {old_checkpoint}: {e}")

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint based on step number."""
    sorted_checkpoints = get_sorted_checkpoints(checkpoint_dir)
    if sorted_checkpoints:
        return sorted_checkpoints[-1]
    return None

# --- Custom Checkpoint Callback ---
class ManagedCheckpointCallback(BaseCallback):
    """
    Custom checkpoint callback that manages disk usage by:
    1. First deleting all existing replay buffers
    2. Then deleting older model checkpoints
    3. Finally saving the current model and replay buffer
    4. Also saves VecNormalize statistics
    5. Also saves curriculum learning state
    """
    
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0, curriculum_manager=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.curriculum_manager = curriculum_manager
        
        # Detect Kaggle environment to avoid disk space issues
        self.is_kaggle = os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    def _on_step(self):
        # Called on every environment step
        if self.n_calls % self.save_freq == 0:
            print(f"\n--- Checkpoint Callback: Saving checkpoint at step {self.num_timesteps} ---")
            
            # Clean up old checkpoints first to save disk space
            cleanup_old_checkpoints(self.save_path)
            
            # Save the current model
            checkpoint_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_path)
            print(f"--- Checkpoint Callback: Model saved to: {checkpoint_path} ---")
            
            # Save the replay buffer (skip on Kaggle due to disk space limits)
            if not self.is_kaggle:
                buffer_path = get_replay_buffer_filename(checkpoint_path)
                self.model.save_replay_buffer(buffer_path)
                print(f"--- Checkpoint Callback: Replay buffer saved to: {buffer_path} ---")
            else:
                print(f"--- Checkpoint Callback: Skipping replay buffer save on Kaggle (disk space limit) ---")
            
            # Save VecNormalize statistics if available
            if hasattr(self.training_env, 'save_running_average'):
                vecnorm_path = get_vecnormalize_filename(checkpoint_path)
                self.training_env.save(vecnorm_path)
                print(f"--- Checkpoint Callback: VecNormalize stats saved to: {vecnorm_path} ---")
            
            # Save curriculum state if available
            if self.curriculum_manager is not None:
                curriculum_path = get_curriculum_filename(checkpoint_path)
                self.curriculum_manager.save(curriculum_path)
                print(f"--- Checkpoint Callback: Curriculum state saved to: {curriculum_path} ---")
            
            # Show disk usage information
            import shutil
            total, used, free = shutil.disk_usage(self.save_path)
            print(f"--- Disk Usage - Total: {total//2**30:.1f}GB, Used: {used//2**30:.1f}GB, Free: {free//2**30:.1f}GB ---")
        
        return True

# --- Custom Callbacks ---
class CurriculumManager:
    """
    Manages workspace size progression based on success rate in curriculum learning.
    
    Attributes:
        config (dict): Configuration dictionary containing curriculum parameters
        vec_env (VecEnv): The vectorized environment to update
        success_history (list): History of recent episode successes (1 for success, 0 for failure)
        env_successes (list): Success counts for each parallel environment
        env_episodes (list): Episode counts for each parallel environment
    """
    def __init__(self, config, vec_env):
        self.config = config
        self.vec_env = vec_env
        self.success_history = []
        self.env_successes = [0] * config['N_ENVS']
        self.env_episodes = [0] * config['N_ENVS']
        self.current_stage = config['CURRENT_CURRICULUM_STAGE']
        self.current_ws_size = config['CURRENT_WS_SIZE']
        
        # Calculate step size for workspace progression
        self.ws_step = (config['FINAL_WS_SIZE'] - config['INITIAL_WS_SIZE']) / max(1, config['CURRICULUM_STEPS'] - 1)
        
        # Initial setup
        self._update_env_workspace()
        
    def _update_env_workspace(self):
        """Update all environments with the current workspace size"""
        half_size = self.current_ws_size / 2
        center_x, center_y = 0.55, 0.0  # Center of the workspace
        
        # Set workspace boundaries for all environments
        low = [center_x - half_size, center_y - half_size, 0.02]
        high = [center_x + half_size, center_y + half_size, 0.02]
        
        # Get the underlying VecEnv (unwrap VecNormalize if present)
        underlying_vec_env = self.vec_env
        if hasattr(self.vec_env, 'venv'):
            # This is VecNormalize, get the underlying environment
            underlying_vec_env = self.vec_env.venv
        
        # Use the public method to set workspace boundaries
        try:
            # For VecEnv, we need to call the method on all environments
            underlying_vec_env.env_method('set_workspace_boundaries', low, high)
            print(f"\n--- Curriculum: Updated workspace size to {self.current_ws_size:.3f}m ---")
            print(f"--- Workspace boundaries: {low} to {high} ---")
        except AttributeError as e:
            # Fallback: Try to access underlying environment directly if it's DummyVecEnv
            print(f"Warning: env_method failed ({e}), trying direct access...")
            try:
                if hasattr(underlying_vec_env, 'envs'):
                    # DummyVecEnv case - direct access to environments
                    for env in underlying_vec_env.envs:
                        # Handle Monitor wrapper
                        actual_env = env.env if hasattr(env, 'env') else env
                        if hasattr(actual_env, 'set_workspace_boundaries'):
                            actual_env.set_workspace_boundaries(low, high)
                    print(f"\n--- Curriculum: Updated workspace size to {self.current_ws_size:.3f}m (direct access) ---")
                    print(f"--- Workspace boundaries: {low} to {high} ---")
                else:
                    print("Error: Cannot access environments to update workspace boundaries")
            except Exception as e2:
                print(f"Error: Could not update workspace boundaries: {e2}")
                print("Continuing with default workspace...")
        except Exception as e:
            print(f"Warning: Could not update workspace boundaries: {e}")
            print("Continuing with default workspace...")
    
    def register_episode_result(self, env_idx, success):
        """
        Register the result of an episode for a specific environment.
        
        Args:
            env_idx (int): The environment index
            success (bool): Whether the episode was successful
        """
        success_int = 1 if success else 0
        self.success_history.append(success_int)
        self.env_successes[env_idx] += success_int
        self.env_episodes[env_idx] += 1
        
        # Limit the history to the evaluation window
        if len(self.success_history) > self.config['EVAL_WINDOW']:
            self.success_history.pop(0)
            
        # Check if we should advance the curriculum
        self._maybe_advance_curriculum()
    
    def _maybe_advance_curriculum(self):
        """Check success rate and advance curriculum if threshold is met"""
        if not self.config['CURRICULUM_ENABLED'] or len(self.success_history) < self.config['EVAL_WINDOW']:
            return
        
        # Calculate success rate over the window
        success_rate = sum(self.success_history) / len(self.success_history)
        
        # Check if we should advance to the next stage
        if success_rate >= self.config['SUCCESS_THRESHOLD'] and self.current_stage < self.config['CURRICULUM_STEPS'] - 1:
            self.current_stage += 1
            
            # Update workspace size
            self.current_ws_size = min(
                self.config['FINAL_WS_SIZE'],
                self.config['INITIAL_WS_SIZE'] + self.ws_step * self.current_stage
            )
            
            # Update config to save with checkpoints
            self.config['CURRENT_CURRICULUM_STAGE'] = self.current_stage
            self.config['CURRENT_WS_SIZE'] = self.current_ws_size
            
            # Clear history for next stage
            self.success_history.clear()
            
            # Update environments
            self._update_env_workspace()
            
            print(f"\n--- Curriculum: Advanced to stage {self.current_stage}/{self.config['CURRICULUM_STEPS']-1} ---")
            print(f"--- Success rate: {success_rate:.2f} >= threshold {self.config['SUCCESS_THRESHOLD']:.2f} ---")
    
    def save(self, path):
        """Save curriculum state to a file"""
        state = {
            'current_stage': self.current_stage,
            'current_ws_size': self.current_ws_size,
            'success_history': self.success_history,
            'env_successes': self.env_successes,
            'env_episodes': self.env_episodes
        }
        np.save(path, state)
        print(f"--- Curriculum state saved to {path} ---")
    
    def load(self, path):
        """Load curriculum state from a file"""
        if not os.path.exists(path):
            print(f"--- No curriculum state file found at {path} ---")
            return False
        
        try:
            state = np.load(path, allow_pickle=True).item()
            self.current_stage = state['current_stage']
            self.current_ws_size = state['current_ws_size']
            self.success_history = state['success_history']
            self.env_successes = state['env_successes']
            self.env_episodes = state['env_episodes']
            
            # Update config
            self.config['CURRENT_CURRICULUM_STAGE'] = self.current_stage
            self.config['CURRENT_WS_SIZE'] = self.current_ws_size
            
            # Update environments
            self._update_env_workspace()
            
            print(f"--- Curriculum state loaded from {path} ---")
            print(f"--- Current stage: {self.current_stage}/{self.config['CURRICULUM_STEPS']-1}, WS size: {self.current_ws_size:.3f}m ---")
            if len(self.success_history) > 0:
                success_rate = sum(self.success_history) / len(self.success_history)
                print(f"--- Current success rate: {success_rate:.2f} (over {len(self.success_history)} episodes) ---")
            return True
        except Exception as e:
            print(f"--- Error loading curriculum state: {e} ---")
            return False

class RewardInfoCallback(BaseCallback):
    """
    Callback to log detailed reward information to terminal.
    """
    def __init__(self, log_freq, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        # List of reward keys to track from the info dict
        self.reward_keys = ['Reward', 'position_reward', #'reach_target_bonus', 
                            'grasp_success_bonus', 'orientation_penalty']
        # Dictionary to store rewards collected since last log
        self._rewards_since_last_log = {key: [] for key in self.reward_keys}
        self._last_log_timestep = 0
        
    def _on_step(self):
        """Called after each step in the environment."""
        # Get current total training steps
        current_total_steps = self.num_timesteps
        
        # Collect information from the infos dictionary returned by env.step()
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info is not None and isinstance(info, dict):
                    for key in self.reward_keys:
                        if key in info:
                            self._rewards_since_last_log[key].append(info[key])
        
        # Check if it's time to log (based on log_freq)
        if current_total_steps >= self._last_log_timestep + self.log_freq:
            print(f"\n--- Reward Info Log @ Timestep ~{current_total_steps} ---")
            for key in self.reward_keys:
                rewards_list = self._rewards_since_last_log[key]
                if rewards_list:
                    mean_reward = np.mean(rewards_list)
                    # Print average to terminal
                    print(f"  {key}_mean: {mean_reward:.4f}")
                    # Also log to SB3 logger (for TensorBoard/log files)
                    self.logger.record(f"reward_info/{key}_mean", mean_reward)
                else:
                    print(f"  {key}_mean: 0.0 (no data)")
                    self.logger.record(f"reward_info/{key}_mean", 0.0)
            print("----------------------------------------------------")
            
            # Reset reward buffer and update last log timestep
            self._rewards_since_last_log = {key: [] for key in self.reward_keys}
            self._last_log_timestep = current_total_steps
            
        # Return True to continue training
        return True

class EvaluationCallback(BaseCallback):
    """
    Callback to periodically evaluate the model on a separate evaluation environment.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1, log_dir=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir
    
    def _on_step(self):
        """Evaluate the model periodically"""
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate the model
            try:
                mean_reward, std_reward = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True
                )
                
                print(f"\n--- Evaluation @ {self.num_timesteps} timesteps ---")
                print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                print("-------------------------------------------")
                
                # Log to tensorboard and SB3 logger
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.log_dir, "best_model.zip")
                    print(f"New best mean reward: {mean_reward:.2f}! Saving to {best_model_path}")
                    self.model.save(best_model_path)
            
            except Exception as e:
                print(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        return True

class CurriculumCallback(BaseCallback):
    """
    Callback to manage curriculum learning during training.
    Monitors episode completions and success rates to adjust difficulty.
    """
    def __init__(self, curriculum_manager, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.episode_start = [True] * len(curriculum_manager.env_successes)
        self.episode_rewards = [0.0] * len(curriculum_manager.env_successes)
    
    def _on_step(self):
        """Monitor episode completions and success rates"""
        # Process info from each environment
        if "dones" in self.locals and "infos" in self.locals:
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            rewards = self.locals["rewards"]
            
            for env_idx, (done, info, reward) in enumerate(zip(dones, infos, rewards)):
                # Track episode rewards
                self.episode_rewards[env_idx] += reward
                
                if done:
                    # Episode completed, check if it was successful
                    success = False
                    if info is not None and isinstance(info, dict):
                        # Success criteria: object was grasped
                        success = info.get('object_grasped', False)
                    
                    # Register result with curriculum manager
                    self.curriculum_manager.register_episode_result(env_idx, success)
                    
                    # Log success info
                    if success:
                        print(f"\n--- Env {env_idx}: Episode successful! Reward: {self.episode_rewards[env_idx]:.2f} ---")
                    else:
                        print(f"\n--- Env {env_idx}: Episode failed. Reward: {self.episode_rewards[env_idx]:.2f} ---")
                    
                    # Reset episode tracking
                    self.episode_rewards[env_idx] = 0.0
        
        return True

# --- MAIN PROCESS ---
if __name__ == '__main__':
    print("--- Main Process: Starting ---")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup configuration based on arguments
    config = setup_config(args)
    
    # Set a fixed seed for better reproducibility
    set_seeds(config['BASE_SEED'])
    print(f"--- Using base seed: {config['BASE_SEED']} ---")
    
    # Output selected configuration
    print("\n--- Configuration: ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-------------------\n")
    
    print(f"--- Using device: {config['DEVICE']} ---")

    # --- Initialize Vectorized Environment ---
    print("--- Main Process: Creating Vectorized Environment ---")
    vec_env = None
    eval_env = None
    
    try:
        # Create list of environment creation functions, each with unique rank and seed
        env_fns = [
            functools.partial(
                make_env, 
                rank=i, 
                seed=config['BASE_SEED'],
                log_dir=config['TENSORBOARD_LOG_DIR']
            )
            for i in range(config['N_ENVS'])
        ]
        
        # Initialize SubprocVecEnv to run environments in separate processes
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        
        # Wrap with VecNormalize for observation and reward normalization
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,           # Normalize observations
            norm_reward=True,        # Normalize rewards
            clip_obs=10.0,          # Clip normalized observations
            clip_reward=10.0,       # Clip normalized rewards
            gamma=0.99,             # Discount factor for reward normalization
            epsilon=1e-8,           # Small constant to avoid division by zero
            training=True           # Enable training mode (updates normalization statistics)
        )
        print(f"--- Main Process: Vectorized Environment with VecNormalize wrapper and {config['N_ENVS']} environments created ---")
        
        # Create a separate environment for evaluation
        eval_env = make_env(
            rank=config['N_ENVS'], 
            seed=config['BASE_SEED']+100,
            log_dir=config['TENSORBOARD_LOG_DIR']
        )
        print("--- Main Process: Evaluation environment created ---")
        
    except Exception as e:
        print(f"!!! Critical error creating environments: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to close VecEnv if it was partially created
        if vec_env is not None:
            print("--- Attempting to close VecEnv after creation error ---")
            try: 
                vec_env.close()
            except Exception as close_e: 
                print(f"Error closing VecEnv during error handling: {close_e}")
        
        exit(1)  # Exit with error code

    # --- Define Policy ---
    # Define neural network architecture
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=dict(features_dim=config['POLICY_FEATURES_DIM']),
        # Can add other policy/value network customizations here
        # net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
    
    # --- Set up callbacks ---
    # Create curriculum manager
    curriculum_manager = CurriculumManager(config, vec_env)
    
    # --- Get latest checkpoint if any ---
    print("--- Main Process: Checking for existing checkpoints ---")
    latest_checkpoint = get_latest_checkpoint(config['CHECKPOINT_DIR'])
    
    # If we found a latest checkpoint, try to load curriculum state
    if latest_checkpoint:
        curriculum_path = get_curriculum_filename(latest_checkpoint)
        curriculum_manager.load(curriculum_path)
    
    # Custom checkpoint callback that manages old checkpoints
    checkpoint_callback = ManagedCheckpointCallback(
        save_freq=max(config['CHECKPOINT_FREQ'] // config['N_ENVS'], 1),  # Adjust for number of environments
        save_path=config['CHECKPOINT_DIR'],
        name_prefix="kuka_pushing_model",
        verbose=1,
        curriculum_manager=curriculum_manager
    )
    
    # Custom callback to log reward information
    reward_info_callback = RewardInfoCallback(
        log_freq=max(config['REWARD_LOG_FREQ'] // config['N_ENVS'], 1),
        verbose=0
    )
    
    # Evaluation callback
    eval_callback = EvaluationCallback(
        eval_env=eval_env,
        eval_freq=max(config['EVAL_FREQ'] // config['N_ENVS'], 1),
        n_eval_episodes=5,
        verbose=1,
        log_dir=config['LOG_DIR_BASE']
    )
    
    # Curriculum callback
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        verbose=0
    )
    
    # Combine callbacks into a list
    callback_list = CallbackList([reward_info_callback, checkpoint_callback, eval_callback, curriculum_callback])

    # --- Load model or Initialize ---
    model = None
    resume_training = False
    actual_buffer_size = config['BUFFER_SIZE']  # Track actual buffer size used
    
    # If we found a latest checkpoint
    if latest_checkpoint:
        try:
            print(f"--- Main Process: Loading latest checkpoint: {latest_checkpoint} onto device '{config['DEVICE']}' ---")
            # Load SAC model from file with replay buffer
            model = SAC.load(
                latest_checkpoint,
                env=vec_env,
                device=config['DEVICE'],
                buffer_size=config['BUFFER_SIZE'],
                # custom_objects can be used if policy_kwargs changed
                # custom_objects={'policy_kwargs': policy_kwargs}
            )
            print(f"--- Main Process: Successfully loaded model from: {latest_checkpoint} ---")
            
            # Check and log information about loaded model
            print(f"--- Model was previously trained for {model.num_timesteps} timesteps ---")
            
            # Load VecNormalize statistics if available
            vecnorm_path = get_vecnormalize_filename(latest_checkpoint)
            if os.path.exists(vecnorm_path):
                try:
                    print(f"--- Loading VecNormalize statistics from: {vecnorm_path} ---")
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    print("--- VecNormalize statistics loaded successfully ---")
                except Exception as e:
                    print(f"Warning: Could not load VecNormalize statistics: {e}")
                    print("--- Continuing with fresh normalization statistics ---")
            else:
                print(f"--- No VecNormalize statistics found at: {vecnorm_path} ---")
                print("--- Continuing with fresh normalization statistics ---")
            
            model.load_replay_buffer(get_replay_buffer_filename(latest_checkpoint))
            
            if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
                # <<< THÊM DÒNG DEBUG NÀY >>>
                print(f"--- DEBUG [LOAD]: Actual Buffer Size RIGHT AFTER LOADING: {model.replay_buffer.buffer_size} ---")
                # <<< KẾT THÚC DÒNG DEBUG >>>
                current_buffer_size = model.replay_buffer.size() if hasattr(model.replay_buffer, 'size') else model.replay_buffer.pos
                print(f"--- Replay buffer contains {current_buffer_size * config['N_ENVS']} / {model.replay_buffer.buffer_size * config['N_ENVS']} samples ---")
                
                # Handle buffer size difference
                if model.replay_buffer.buffer_size * config['N_ENVS'] != config['BUFFER_SIZE']:
                    print(f"Warning: Saved buffer size ({model.replay_buffer.buffer_size}) differs from current config ({config['BUFFER_SIZE']}).")
                    print(f"Using loaded buffer size for consistency.")
                    actual_buffer_size = model.replay_buffer.buffer_size
            else:
                print("Warning: Replay buffer not found or not loaded from the file.")
                print("Starting with an empty replay buffer.")
                
            resume_training = True
            
        except Exception as e:
            # Handle error if model can't be loaded
            print(f"!!! Error loading model from {latest_checkpoint}: {e}")
            import traceback
            traceback.print_exc()
            print("--- Main Process: Could not load the existing model. Will start training from scratch. ---")
            model = None
            resume_training = False
    
    # If no model was loaded (due to error or fresh start)
    if model is None:
        print("--- Main Process: Initializing new SAC model ---")
        # Initialize new SAC model
        model = SAC(
            "MultiInputPolicy",     # Policy for diverse inputs (image + state)
            vec_env,                # Vectorized environment
            policy_kwargs=policy_kwargs,  # Neural network architecture parameters
            buffer_size=actual_buffer_size,  # Replay buffer size
            learning_starts=config['LEARNING_STARTS'],  # Steps before starting network updates
            learning_rate=config['LEARNING_RATE'],  # Learning rate for optimizer
            batch_size=config['BATCH_SIZE'],  # Batch size for each update
            verbose=1,              # Log info level (1: basic info)
            tensorboard_log=config['TENSORBOARD_LOG_DIR'],  # TensorBoard log directory
            device=config['DEVICE']    # Computing device (CPU/GPU)
        )
        print(f"--- New SAC model initialized with buffer size {actual_buffer_size}, learning starts {config['LEARNING_STARTS']} ---")

        # <<< THÊM DÒNG DEBUG NÀY >>>
        if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
            print(f"--- DEBUG [INIT]: Actual Replay Buffer Size RIGHT AFTER INIT: {model.replay_buffer.buffer_size} ---")
        else:
            print("--- DEBUG [INIT]: Replay buffer not found immediately after init! ---")
        # <<< KẾT THÚC DÒNG DEBUG >>>

        resume_training = False

    # --- Training ---
    learn_kwargs = {}
    # Determine whether to reset timestep counter or not
    if resume_training:
        print(f"--- Main Process: Resuming training from {model.num_timesteps} up to {config['TOTAL_TIMESTEPS']} timesteps ---")
        # Don't reset timestep counter when continuing training
        learn_kwargs["reset_num_timesteps"] = False
    else:
        print(f"--- Main Process: Starting new training up to {config['TOTAL_TIMESTEPS']} timesteps ---")
        # Reset timestep counter for new training
        learn_kwargs["reset_num_timesteps"] = True
    
    try:
        # Start (or continue) training process
        print(f"\n--- Main Process: Starting model.learn() with reset_num_timesteps={learn_kwargs.get('reset_num_timesteps', True)} ---")
        print(f"--- Logging SB3 defaults every {config['LOG_INTERVAL']} steps ---")
        print(f"--- Logging custom rewards every {config['REWARD_LOG_FREQ']} steps ---")
        print(f"--- Saving checkpoints every {config['CHECKPOINT_FREQ']} steps ---")
        print(f"--- Evaluating model every {config['EVAL_FREQ']} steps ---")
        
        model.learn(
            total_timesteps=config['TOTAL_TIMESTEPS'],  # Total training steps
            callback=callback_list,           # List of callbacks to use
            log_interval=config['LOG_INTERVAL'],        # SB3 default logging frequency
    
            **learn_kwargs                    # Other parameters (like reset_num_timesteps)
        )
        
        # Training completed successfully
        print(f"\n--- Main Process: Training finished successfully (reached {model.num_timesteps} / {config['TOTAL_TIMESTEPS']} timesteps). ---")
        print(f"--- Saving final model to: {config['FINAL_MODEL_PATH']} ---")
        model.save(config['FINAL_MODEL_PATH'])
        
    except KeyboardInterrupt:
        # Handle when user presses Ctrl+C
        print("\n--- Main Process: Training interrupted by user (KeyboardInterrupt) ---")
        print("--- The latest checkpoint has already been saved by the callback ---")
        print("--- Run the script again to resume from the latest checkpoint ---")
    
    except Exception as e:
        # Handle other errors during training
        print(f"\n--- Main Process: An unexpected error occurred during training: {e} ---")
        import traceback
        traceback.print_exc()
        print("--- The latest checkpoint should be available from the last successful save ---")
    
    finally:
        # --- Cleanup (Always executed) ---
        print("\n--- Main Process: Cleaning up resources ---")
        try:
            # Make sure to close vectorized environment to free resources
            if vec_env is not None:
                print("--- Closing VecEnv... ---")
                vec_env.close()
                print("--- VecEnv closed successfully ---")
            
            # Also close evaluation environment if created
            if eval_env is not None:
                print("--- Closing evaluation environment... ---")
                eval_env.close()
                print("--- Evaluation environment closed successfully ---") 
        except Exception as close_e:
            # Log error if can't close environments
            print(f"Error closing environments during cleanup: {close_e}")
        
        print("--- Main Process: Script finished ---")