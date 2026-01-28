import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")   

devNumber = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
print(f"Device number: {devNumber}")

devName = torch.cuda.get_device_name(devNumber) if torch.cuda.is_available() else 'cpu' 
print(f"Device name: {devName}")

# create a tensor on cpu

T1=torch.rand(4,4)
print("CPU Tensor:")
print(T1)

# this will convert CPU tensor to GPU tensor
if torch.cuda.is_available():
    T2=T1.to(device)
    print("GPU Tensor:")
    print(T2)