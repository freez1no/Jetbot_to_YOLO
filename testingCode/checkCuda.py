"""

CUDA 사용 여부를 체크하기위한 코드입니다.
This code for tesing CUDA availability.

"""

import torch
torch.cuda.init()

if torch.cuda.is_available():
    print(f"CUDA Device Name : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count : {torch.cuda.device_count()}")
    print(f"CUDA Current Device Index : {torch.cuda.current_device()}")
    print(f"CUDA Memory Allocated : {torch.cuda.memory_allocated(device=None)} bytes")
    print(f"CUDA Memory Cached : {torch.cuda.memory_reserved(device=None)} bytes")
else:
    print("CUDA is not available.")

a = input("Reset peak memory stats? (y/n): ")
if a.lower() == 'y':
    torch.cuda.reset_peak_memory_stats(device=None)
    print("Peak memory stats reset.")