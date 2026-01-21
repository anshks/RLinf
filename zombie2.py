import torch
import time
import threading

# Global dictionary to keep tensors alive across iterations
# Key: device index (int), Value: tensor
_zombie_tensors = {}
_zombie_initialized = False
_background_threads = []
_stop_background = False

# You can tune this if you want
BLOCK_SIZE = 32          # smaller tile is lighter
COMPUTE_ROUNDS = 800       # reduce matmul rounds per burst
DUTY_CYCLE_SLEEP = 0.01   # sleep between bursts to avoid 100% util


def initialize_zombie_operation(target_percent=0.05, verbose=True):
    """
    Initialize large random tensors on ALL available GPUs.

    Args:
        target_percent: Percentage of GPU memory to occupy (0.0 to 1.0).
                        Default 0.05 (5%).
    """
    global _zombie_tensors, _zombie_initialized

    if _zombie_initialized:
        if verbose:
            print("Zombie tensors already initialized, skipping...")
        return

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA is not available. Zombie operation disabled.")
        return

    num_gpus = torch.cuda.device_count()
    if verbose:
        print(f"[ZOMBIE] Found {num_gpus} GPUs.")

    for i in range(num_gpus):
        try:
            device = torch.device(f"cuda:{i}")

            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory

            # Target memory usage
            target_memory = int(total_memory * target_percent)

            if verbose:
                print(
                    f"[ZOMBIE] GPU {i}: Total {total_memory / (1024**3):.2f} GB, "
                    f"Target {target_memory / (1024**3):.2f} GB ({target_percent*100:.1f}%)"
                )

            # Calculate tensor size for float32 (4 bytes per element)
            num_elements = target_memory // 4

            # Create a large square-ish tensor for better memory layout
            side_length = int(num_elements**0.5)

            if verbose:
                print(
                    f"[ZOMBIE] GPU {i}: Allocating tensor of shape ({side_length}, {side_length})..."
                )

            # Create the large random tensor on the specific device
            _zombie_tensors[i] = torch.randn(
                side_length, side_length, device=device, dtype=torch.float32
            )

            if verbose:
                allocated = torch.cuda.memory_allocated(i)
                print(f"[ZOMBIE] GPU {i}: Allocated {allocated / (1024**3):.2f} GB")

        except Exception as e:
            print(f"[ZOMBIE] GPU {i}: Failed to initialize: {e}")

    _zombie_initialized = True


def run_zombie_operation(device_id=None, verbose=False):
    """
    Heavier, continuous compute on zombie tensors to keep GPU util high.

    If device_id is None, runs on all initialized GPUs (sequentially).
    """
    global _zombie_tensors

    if device_id is not None:
        devices_to_run = [device_id]
    else:
        devices_to_run = list(_zombie_tensors.keys())

    for dev in devices_to_run:
        if dev not in _zombie_tensors:
            continue

        tensor = _zombie_tensors[dev]

        # Choose a reasonably large block to drive utilization.
        side = min(BLOCK_SIZE, tensor.shape[0], tensor.shape[1])
        block = tensor[:side, :side]

        # Few back-to-back matmuls + elementwise ops
        # This is intentionally heavier than before to push avg util > 60%.
        for _ in range(COMPUTE_ROUNDS):
            tmp = torch.matmul(block, block)  # heavy
            block.add_(0.000001 * tmp)        # in-place update to avoid new allocs
            block.tanh_()                     # elementwise nonlinearity

        if verbose:
            allocated_memory = torch.cuda.memory_allocated(dev)
            print(
                f"[ZOMBIE] GPU {dev} Memory: {allocated_memory / (1024**3):.2f} GB "
                f"(block {side}x{side})"
            )


def _background_compute_worker(device_id):
    """
    Worker thread for a specific GPU.
    Runs continuous heavy-ish operations to keep utilization high.
    """
    global _stop_background, _zombie_tensors

    torch.cuda.set_device(device_id)

    while not _stop_background:
        if device_id not in _zombie_tensors:
            break
        try:
            # Continuous compute with a brief sleep to lower duty cycle.
            run_zombie_operation(device_id=device_id, verbose=False)
            # Ensure GPU work completes so nvidia-smi reports accurately
            torch.cuda.synchronize(device_id)
            if DUTY_CYCLE_SLEEP > 0:
                time.sleep(DUTY_CYCLE_SLEEP)
        except Exception as e:
            print(f"[ZOMBIE] Background worker GPU {device_id} error: {e}")
            break


def start_background_compute(verbose=True):
    """
    Start background threads for ALL initialized GPUs.
    """
    global _background_threads, _stop_background, _zombie_initialized

    if len(_background_threads) > 0:
        if verbose:
            print("[ZOMBIE] Background compute already running")
        return

    if not _zombie_initialized:
        initialize_zombie_operation(verbose=verbose)

    _stop_background = False

    # Start one thread per GPU
    for device_id in _zombie_tensors.keys():
        t = threading.Thread(
            target=_background_compute_worker,
            args=(device_id,),
            daemon=True,
        )
        t.start()
        _background_threads.append(t)

    if verbose:
        print(
            f"[ZOMBIE] Started {len(_background_threads)} background compute "
            f"threads (one per GPU)."
        )


def stop_background_compute(verbose=True):
    """
    Stop all background compute threads.
    """
    global _background_threads, _stop_background

    if not _background_threads:
        return

    _stop_background = True
    for t in _background_threads:
        if t.is_alive():
            t.join(timeout=2.0)

    _background_threads = []

    if verbose:
        print("[ZOMBIE] Stopped background compute threads")


def cleanup_zombie_operation(verbose=True):
    """
    Clean up and free memory on all GPUs.
    """
    global _zombie_tensors, _zombie_initialized

    stop_background_compute(verbose=False)

    if _zombie_tensors:
        if verbose:
            print("[ZOMBIE] Cleaning up zombie tensors...")

        _zombie_tensors.clear()
        _zombie_initialized = False
        torch.cuda.empty_cache()

        if verbose:
            print("[ZOMBIE] Memory freed.")


def create_large_tensor_operation():
    """
    Standalone version.
    """
    # Initialize with a lighter 2% memory target
    initialize_zombie_operation(target_percent=0.02, verbose=True)

    if not torch.cuda.is_available():
        return

    print("\n[ZOMBIE] Starting background compute on all GPUs...")
    start_background_compute()  # no sleep inside worker

    print("\n[ZOMBIE] Running. Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup_zombie_operation(verbose=True)


if __name__ == "__main__":
    create_large_tensor_operation()