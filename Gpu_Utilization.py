import torch
import time

def main():
    # Check if CUDA is available and if GPU 1 is available
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("CUDA is not available or GPU 1 is not available. Exiting...")
        return

    # Explicitly set the device to GPU 1
    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.get_device_name(device)}")

    # Define the size of the square matrices and create them in GPU 1's memory
    n = 10000
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    e = torch.randn(n, n, device=device)
    f = torch.randn(n, n, device=device)

    # Determine the end time (current time + 5 minutes)
    end_time = time.time() + 20 * 60  # 5 minutes from now

    # Loop to keep performing matrix multiplication until 5 minutes have passed
    iterations = 0
    while time.time() < end_time:
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        d = torch.matmul(c, e)
        g = torch.matmul(d, f)
        h = torch.matmul(g, b)
        i = torch.matmul(h, e)
        j = torch.matmul(i, f)
        k = torch.matmul(j, b)
        l = torch.matmul(k, e)
        m = torch.matmul(l, f)
        o = torch.matmul(m, e)
        p = torch.matmul(o, f)
        q = torch.matmul(p, b)
        r = torch.matmul(q, e)
        s = torch.matmul(r, f)
        t = torch.matmul(s, b)
        # u = torch.matmul(t, e)
        # v = torch.matmul(u, f)
        # w = torch.matmul(v, b)
        # x = torch.matmul(w, e)
        # y = torch.matmul(x, f)
        # z = torch.matmul(y, b)
        iterations += 1

    print(f"Completed {iterations} iterations of matrix multiplication in 5 minutes.")

if __name__ == "__main__":
    main()
