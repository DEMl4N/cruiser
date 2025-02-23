import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2
import os

from neural_engine import NeuralEngine


class ODEngine(NeuralEngine):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    # class HostDeviceMem:
    # 	def __init__(self, host_mem, device_mem):
    # 		self.host = host_mem
    # 		self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, allocations = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True

            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)

            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s

            # Allocate host and device buffers
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }

            # Append to the appropiate input/output list
            allocations.append(allocation)
            if self.engine.binding_is_input(i):  # 🔹 get_tensor_mode() 대신 사용
                inputs.append(binding)
            else:
                outputs.append(binding)
        return inputs, outputs, allocations, stream

    def infer(self, input_image):
        # Transfer input data to device
        image = input_image.transpose(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        image = np.ascontiguousarray(image)
        cuda.memcpy_htod_async(self.inputs[0]['allocation'], image, self.stream)
        self.context.execute_async_v2(self.allocations, stream_handle=self.stream.handle)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'], self.stream)

    def extract_boxes_and_classes(self, output_tensor, conf_threshold=0.5):
        # classes_id, boxes, scores = [], [], []
        object_info = []

        # Extract the bounding box, objectness, and class probabilities
        bbox = output_tensor[:4]  # [x_center, y_center, width, height]
        probs = output_tensor[4:]  # (0th class probs[], 1st class probs[], ...)

        for cls, prob in enumerate(probs):
            target_col = np.argmax(prob)  # Index of the most likely bbox of each car

            print(f"class {cls} conf : {prob[target_col]}")
            if prob[target_col] >= conf_threshold:
                box = bbox[:, target_col]
                conf = prob[target_col]

                object_info.append((cls, box, conf))

        return object_info

    def get_output(self):
        self.stream.synchronize()

        result = self.outputs[0]['host_allocation'][0]
        return self.extract_boxes_and_classes(result)
