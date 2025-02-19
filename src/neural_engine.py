import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

class NeuralEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self.engine = None
        self.context = None
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()

        if self.engine is None or self.context is None:
            print("None..")
            exit()

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem


    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(engine.num_bindings):
            tensor_name = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))
            # Append to the appropiate input/output list
            if self.engine.binding_is_input(i):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
            return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i].host, self.outputs[i].device, self.stream)

    def get_output(self):
        self.stream.synchronize()

        return [self.outputs[i].host.tolist() for i in range(len(self.outputs))]