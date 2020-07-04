# tmp_issue
Details about failures in the test of hugginface.   
```
================================================================================================================================ FAILURES ================================================================================================================================
__________________________________________________________________________________________________________ ElectraModelTest.test_multigpu_data_parallel_forward _________________________________________________________________________________________________________$
[gw16] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_electra.ElectraModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            with torch.no_grad():
>               _ = model(**self._prepare_for_class(inputs_dict, model_class))

tests/test_modeling_common.py:814:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:153: in forward
    return self.gather(outputs, self.output_device)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:165: in gather
    return gather(outputs, output_device, dim=self.dim)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:68: in gather
    res = gather_map(outputs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:63: in gather_map
    return type(out)(map(gather_map, zip(*outputs)))
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:55: in gather_map
    return Gather.apply(target_device, dim, *outputs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:68: in forward
    return comm.gather(inputs, ctx.dim, ctx.target_device)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tensors = (tensor([[-0.0003, -0.0024, -0.0031, -0.0139, -0.0099,  0.0039,  0.0020],
        [ 0.0005,  0.0050,  0.0051, -0.0022,...     device='cuda:2'), tensor([ 0.0064, -0.0016,  0.0008,  0.0008, -0.0058, -0.0016, -0.0039],
       device='cuda:3')), dim = 0
destination = 0

    def gather(tensors, dim=0, destination=None):
        """Gathers tensors from multiple GPUs.

        Tensor sizes in all dimension different than ``dim`` have to match.

        Arguments:
            tensors (Iterable[Tensor]): iterable of tensors to gather.
            dim (int): a dimension along which the tensors will be concatenated.
            destination (int, optional): output device (-1 means CPU, default:
                current device)

        Returns:
            A tensor located on ``destination`` device, that is a result of
            concatenating ``tensors`` along ``dim``.
        """
>       return torch._C._gather(tensors, dim, destination)
E       RuntimeError: tensor.ndimension() == static_cast<int64_t>(expected_size.size()) INTERNAL ASSERT FAILED at /pytorch/torch/csrc/cuda/comm.cpp:225, please report a bug to PyTorch.  (gather at /pytorch/torch/csrc/cuda/comm.cpp:225)
E       frame #0: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x33 (0x7ff618cfa193 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libc10.so)
E       frame #1: torch::cuda::gather(c10::ArrayRef<at::Tensor>, long, c10::optional<int>) + 0x856 (0x7ff61dd0d856 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #2: <unknown function> + 0x9d646d (0x7ff66486446d in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #3: <unknown function> + 0x2961c4 (0x7ff6641241c4 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #4: _PyCFunction_FastCallKeywords + 0x1eb (0x523d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #5: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e1f9]
E       frame #6: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #7: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #8: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #9: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #10: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #11: PyEval_EvalCodeEx + 0x5b2 (0x57ea82 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #12: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x4fc083]
E       frame #13: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #14: THPFunction_apply(_object*, _object*) + 0xa8f (0x7ff6644f482f in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #15: PyCFunction_Call + 0x5f (0x52422f in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #16: _PyEval_EvalFrameDefault + 0x606f (0x57c5bf in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #17: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575e74]
E       frame #18: _PyFunction_FastCallDict + 0x1da (0x57fbda in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #19: _PyObject_FastCallDict + 0x1d9 (0x4e7b19 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #20: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57428d]
E       frame #21: PySequence_Tuple + 0xd8 (0x4e7488 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #22: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x52ed50]
E       frame #23: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x533875]
E       frame #24: _PyObject_FastCallKeywords + 0x10b (0x4e7d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #25: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e385]
E       frame #26: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #27: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575e74]
E       frame #28: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #29: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #30: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #31: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575dce]
E       frame #32: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #33: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #34: _PyEval_EvalFrameDefault + 0x11aa (0x5776fa in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #35: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #36: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #37: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #38: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #39: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #40: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #41: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #42: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #43: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #44: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #45: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #46: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #47: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x539564]
E       frame #48: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #49: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #50: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #51: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #52: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #53: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #54: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #55: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #56: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #57: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #58: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #59: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #60: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #61: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #62: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x539564]
E       frame #63: _PyObject_FastCallKeywords + 0x10b (0x4e7d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/cuda/comm.py:165: RuntimeError
___________________________________________________________________________________________________________ GPT2ModelTest.test_multigpu_data_parallel_forward __________________________________________________________________________________________________[529/1877]
[gw19] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_gpt2.GPT2ModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            with torch.no_grad():
>               _ = model(**self._prepare_for_class(inputs_dict, model_class))

tests/test_modeling_common.py:814:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:153: in forward
    return self.gather(outputs, self.output_device)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:165: in gather
    return gather(outputs, output_device, dim=self.dim)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:68: in gather
    res = gather_map(outputs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:63: in gather_map
    return type(out)(map(gather_map, zip(*outputs)))
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:63: in gather_map
    return type(out)(map(gather_map, zip(*outputs)))
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py:55: in gather_map
    return Gather.apply(target_device, dim, *outputs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:68: in forward
    return comm.gather(inputs, ctx.dim, ctx.target_device)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tensors = (tensor([[[[[ 1.0881e-01, -3.3284e-02,  1.2725e-02,  ...,  4.3066e-02,
             1.3836e-01,  3.4084e-02],
        ...4e-02, -6.6636e-02,  8.0582e-02, -1.1744e-03,
             1.7088e-01, -3.0254e-02, -1.4164e-01]]]]], device='cuda:3')), dim = 0
destination = 0

    def gather(tensors, dim=0, destination=None):
        """Gathers tensors from multiple GPUs.

        Tensor sizes in all dimension different than ``dim`` have to match.

        Arguments:
            tensors (Iterable[Tensor]): iterable of tensors to gather.
            dim (int): a dimension along which the tensors will be concatenated.
            destination (int, optional): output device (-1 means CPU, default:
                current device)

        Returns:
            A tensor located on ``destination`` device, that is a result of
            concatenating ``tensors`` along ``dim``.
        """
>       return torch._C._gather(tensors, dim, destination)
E       RuntimeError: Gather got an input of invalid size: got [2, 2, 4, 7, 8], but expected [2, 4, 4, 7, 8] (gather at /pytorch/torch/csrc/cuda/comm.cpp:231)
E       frame #0: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x33 (0x7ff7d4df2193 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libc10.so)
E       frame #1: torch::cuda::gather(c10::ArrayRef<at::Tensor>, long, c10::optional<int>) + 0x2ad (0x7ff7d9e052ad in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #2: <unknown function> + 0x9d646d (0x7ff82095c46d in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #3: <unknown function> + 0x2961c4 (0x7ff82021c1c4 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #4: _PyCFunction_FastCallKeywords + 0x1eb (0x523d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #5: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e1f9]
E       frame #6: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #7: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #8: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #9: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #10: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #11: PyEval_EvalCodeEx + 0x5b2 (0x57ea82 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #12: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x4fc083]
E       frame #13: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #14: THPFunction_apply(_object*, _object*) + 0xa8f (0x7ff8205ec82f in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #15: PyCFunction_Call + 0x5f (0x52422f in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #16: _PyEval_EvalFrameDefault + 0x606f (0x57c5bf in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #17: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575e74]
E       frame #18: _PyFunction_FastCallDict + 0x1da (0x57fbda in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #19: _PyObject_FastCallDict + 0x1d9 (0x4e7b19 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #20: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57428d]
E       frame #21: PySequence_Tuple + 0xd8 (0x4e7488 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #22: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x52ed50]
E       frame #23: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x533875]
E       frame #24: _PyObject_FastCallKeywords + 0x10b (0x4e7d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #25: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e385]
E       frame #26: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #27: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575e74]
E       frame #28: _PyFunction_FastCallDict + 0x1da (0x57fbda in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #29: _PyObject_FastCallDict + 0x1d9 (0x4e7b19 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #30: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57428d]
E       frame #31: PySequence_Tuple + 0x137 (0x4e74e7 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #32: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x52ed50]
E       frame #33: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x533875]
E       frame #34: _PyObject_FastCallKeywords + 0x10b (0x4e7d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #35: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e385]
E       frame #36: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #37: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575e74]
E       frame #38: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #39: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #40: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #41: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575dce]
E       frame #42: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #43: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #44: _PyEval_EvalFrameDefault + 0x11aa (0x5776fa in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #45: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #46: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #47: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #48: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #49: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #50: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #51: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #52: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #53: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #54: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #55: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #56: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #57: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x539564]
E       frame #58: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #59: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #60: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #61: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #62: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #63: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/cuda/comm.py:165: RuntimeError
_____________________________________________________________________________________________________________________ BartHeadTests.test_lm_forward ______________________________________________________________________________________________________________________
[gw9] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_bart.BartHeadTests testMethod=test_lm_forward>

    @timeout_decorator.timeout(1)
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size).to(torch_device)
        lm_model = BartForConditionalGeneration(config)
        lm_model.to(torch_device)
>       loss, logits, enc_features = lm_model(input_ids=input_ids, labels=lm_labels)

tests/test_modeling_bart.py:413:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
src/transformers/modeling_bart.py:1003: in forward
    output_hidden_states=output_hidden_states,
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
src/transformers/modeling_bart.py:876: in forward
    use_cache=use_cache,
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
src/transformers/modeling_bart.py:531: in forward
    output_attentions=output_attentions,
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
src/transformers/modeling_bart.py:407: in forward
    x = self.encoder_attn_layer_norm(x)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/apex/normalization/fused_layer_norm.py:156: in forward
    input, self.normalized_shape, self.weight, self.bias, self.eps)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1696: in layer_norm
    torch.backends.cudnn.enabled)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/timeout_decorator/timeout_decorator.py:72: in handler
    _raise_exception(timeout_exception, exception_message)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
exception = <class 'timeout_decorator.timeout_decorator.TimeoutError'>, exception_message = None

    def _raise_exception(exception, exception_message):
        """ This function checks if a exception message is given.

        If there is no exception message, the default behaviour is maintained.
        If there is an exception message, the message is passed to the exception with the 'value' keyword.
        """
        if exception_message is None:
>           raise exception()
E           timeout_decorator.timeout_decorator.TimeoutError: 'Timed Out'

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/timeout_decorator/timeout_decorator.py:45: TimeoutError
___________________________________________________________________________________________________________ CTRLModelTest.test_multigpu_data_parallel_forward ____________________________________________________________________________________________________________
[gw15] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_ctrl.CTRLModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
>               inputs_dict[k] = v.to(0)
E               RuntimeError: CUDA error: out of memory

tests/test_modeling_common.py:804: RuntimeError
[gw2] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_xlnet.XLNetModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
>               inputs_dict[k] = v.to(0)
E               RuntimeError: CUDA error: out of memory

tests/test_modeling_common.py:804: RuntimeError
_________________________________________________________________________________________________________ OpenAIGPTModelTest.test_multigpu_data_parallel_forward _________________________________________________________________________________________________________
[gw23] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_openai.OpenAIGPTModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            with torch.no_grad():
>               _ = model(**self._prepare_for_class(inputs_dict, model_class))

tests/test_modeling_common.py:814:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/modules/module.py:532: in __call__
    result = self.forward(*input, **kwargs)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:151: in forward
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py:156: in replicate
    return replicate(module, device_ids, not torch.is_grad_enabled())
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/replicate.py:103: in replicate
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/parallel/replicate.py:67: in _broadcast_coalesced_reshape
    return comm.broadcast_coalesced(tensors, devices)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tensors = [tensor([[[[1., 0., 0.,  ..., 0., 0., 0.],
          [1., 1., 0.,  ..., 0., 0., 0.],
          [1., 1., 1.,  ..., 0., ..., 1., 0., 0.],
          [1., 1., 1.,  ..., 1., 1., 0.],
          [1., 1., 1.,  ..., 1., 1., 1.]]]], device='cuda:0')]
devices = [0, 1, 2, 3], buffer_size = 10485760

    def broadcast_coalesced(tensors, devices, buffer_size=10485760):
        """Broadcasts a sequence tensors to the specified GPUs.
        Small tensors are first coalesced into a buffer to reduce the number
        of synchronizations.

        Arguments:
            tensors (sequence): tensors to broadcast.
            devices (Iterable): an iterable of devices among which to broadcast.
              Note that it should be like (src, dst1, dst2, ...), the first element
              of which is the source device to broadcast from.
            buffer_size (int): maximum size of the buffer used for coalescing

        Returns:
            A tuple containing copies of the ``tensor``, placed on devices
            corresponding to indices from ``devices``.
        """
>       return torch._C._broadcast_coalesced(tensors, devices, buffer_size)
E       RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 15.78 GiB total capacity; 5.33 MiB already allocated; 17.19 MiB free; 6.00 MiB reserved in total by PyTorch) (malloc at /pytorch/c10/cuda/CUDACachingAllocator.cpp:289)
E       frame #0: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x33 (0x7fa44c250193 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libc10.so)
E       frame #1: <unknown function> + 0x1bccc (0x7fa44c491ccc in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libc10_cuda.so)
E       frame #2: <unknown function> + 0x1cd5e (0x7fa44c492d5e in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libc10_cuda.so)
E       frame #3: THCStorage_resize + 0xa3 (0x7fa450f566f3 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #4: THCTensor_resizeNd + 0x3e9 (0x7fa450f69eb9 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #5: THCudaTensor_catArray + 0x376 (0x7fa45128f9c6 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #6: <unknown function> + 0x464c1ef (0x7fa450ef71ef in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #7: <unknown function> + 0x460aeb5 (0x7fa450eb5eb5 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #8: <unknown function> + 0x1f50aa9 (0x7fa44e7fbaa9 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #9: at::native::cat(c10::ArrayRef<at::Tensor>, long) + 0x23e (0x7fa44e5b2c3e in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #10: <unknown function> + 0x200ef0a (0x7fa44e8b9f0a in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #11: <unknown function> + 0x1f50aa9 (0x7fa44e7fbaa9 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #12: <unknown function> + 0x3d66174 (0x7fa450611174 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #13: <unknown function> + 0x1f50aa9 (0x7fa44e7fbaa9 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #14: torch::cuda::broadcast_coalesced(c10::ArrayRef<at::Tensor>, c10::ArrayRef<long>, unsigned long) + 0xae8 (0x7fa45125f828 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch.so)
E       frame #15: <unknown function> + 0x9d6824 (0x7fa497dba824 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #16: <unknown function> + 0x2961c4 (0x7fa49767a1c4 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
E       frame #17: _PyCFunction_FastCallKeywords + 0x1eb (0x523d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #18: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e1f9]
E       frame #19: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #20: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #21: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #22: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #23: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #24: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575dce]
E       frame #25: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #26: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #27: _PyEval_EvalFrameDefault + 0x11aa (0x5776fa in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #28: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575dce]
E       frame #29: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f31b]
E       frame #30: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #31: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #32: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #33: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #34: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #35: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #36: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #37: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #38: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #39: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #40: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #41: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #42: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #43: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #44: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x539564]
E       frame #45: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #46: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #47: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]
E       frame #48: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e2dc]
E       frame #49: _PyEval_EvalFrameDefault + 0x460 (0x5769b0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #50: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #51: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #52: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #53: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #54: _PyEval_EvalFrameDefault + 0x1a60 (0x577fb0 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #55: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x575a9f]
E       frame #56: _PyFunction_FastCallDict + 0x440 (0x57fe40 in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #57: _PyObject_Call_Prepend + 0x24c (0x4e8b5c in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #58: PyObject_Call + 0x3a (0x4e81ea in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #59: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x539564]
E       frame #60: _PyObject_FastCallKeywords + 0x10b (0x4e7d0b in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #61: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57e385]
E       frame #62: _PyEval_EvalFrameDefault + 0x11aa (0x5776fa in /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python)
E       frame #63: /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python() [0x57f25d]

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/cuda/comm.py:39: RuntimeError
________________________________________________________________________________________________________ MobileBertModelTest.test_multigpu_data_parallel_forward _________________________________________________________________________________________________________
[gw22] linux -- Python 3.6.11 /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/bin/python

self = <tests.test_modeling_mobilebert.MobileBertModelTest testMethod=test_multigpu_data_parallel_forward>

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
>               inputs_dict[k] = v.to(0)
E               RuntimeError: CUDA error: out of memory

tests/test_modeling_common.py:804: RuntimeError
============================================================================================================================ warnings summary ============================================================================================================================
src/transformers/modeling_auto.py:798
src/transformers/modeling_auto.py:798
src/transformers/modeling_auto.py:798
src/transformers/modeling_auto.py:798
src/transformers/modeling_auto.py:798
src/transformers/modeling_auto.py:798
  /home/msrauser/project/transformers/src/transformers/modeling_auto.py:798: FutureWarning: The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` fo
r masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models.
    FutureWarning,

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122
  /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:122: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in t
he opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224
  /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:224: UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
    warnings.warn("To get the last learning rate computed by the scheduler, "
src/transformers/tokenization_utils_base.py:1490: 1099 tests with warnings
  /home/msrauser/project/transformers/src/transformers/tokenization_utils_base.py:1490: DeprecationWarning: The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can
give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='
only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).
    DeprecationWarning,

src/transformers/tokenization_utils_base.py:1464: 209 tests with warnings
  /home/msrauser/project/transformers/src/transformers/tokenization_utils_base.py:1464: DeprecationWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequen
ce in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
    DeprecationWarning,

src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
  /home/msrauser/project/transformers/src/transformers/modeling_gpt2.py:149: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    w = w / (float(v.size(-1)) ** 0.5)
src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
src/transformers/modeling_gpt2.py:149
  /home/msrauser/project/transformers/src/transformers/modeling_gpt2.py:149: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    w = w / (float(v.size(-1)) ** 0.5)

src/transformers/modeling_gpt2.py:151
src/transformers/modeling_gpt2.py:151
src/transformers/modeling_gpt2.py:151
src/transformers/modeling_gpt2.py:151
  /home/msrauser/project/transformers/src/transformers/modeling_gpt2.py:151: TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    mask = self.bias[:, :, ns - nd : ns, :ns]

tests/test_tokenization_fast.py:800
tests/test_tokenization_fast.py:800
tests/test_tokenization_fast.py:800
tests/test_tokenization_fast.py:800
  /home/msrauser/project/transformers/tests/test_tokenization_fast.py:800: DeprecationWarning: Please use assertEqual instead.
    self.assertEquals(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

tests/test_tokenization_fast.py:805
tests/test_tokenization_fast.py:805
tests/test_tokenization_fast.py:805
tests/test_tokenization_fast.py:805
  /home/msrauser/project/transformers/tests/test_tokenization_fast.py:805: DeprecationWarning: Please use assertEqual instead.
    sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]),

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/_reduction.py:13
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/_reduction.py:13
  /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/_reduction.py:13: UserWarning: reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.
    warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")

HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468
  /home/msrauser/project/transformers/HuggingFaceEnv-py36-torch140-cu101/lib/python3.6/site-packages/torch/nn/functional.py:1468: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python va
lues, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    assert padding_idx < weight.size(0), 'Padding_idx must be within num_embeddings'
src/transformers/modeling_flaubert.py:175
src/transformers/modeling_flaubert.py:175
src/transformers/modeling_flaubert.py:175
  /home/msrauser/project/transformers/src/transformers/modeling_flaubert.py:175: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant
 in the future. This means that the trace might not generalize to other inputs!
    assert lengths.size(0) == bs

src/transformers/modeling_flaubert.py:176
src/transformers/modeling_flaubert.py:176
src/transformers/modeling_flaubert.py:176
  /home/msrauser/project/transformers/src/transformers/modeling_flaubert.py:176: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant
in the future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen

src/transformers/modeling_flaubert.py:176
src/transformers/modeling_flaubert.py:176
src/transformers/modeling_flaubert.py:176
  /home/msrauser/project/transformers/src/transformers/modeling_flaubert.py:176: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant
 in the future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen

src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:76: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the
 future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen

src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
src/transformers/modeling_xlm.py:76
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:76: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen
src/transformers/modeling_xlm.py:87
src/transformers/modeling_xlm.py:87
src/transformers/modeling_xlm.py:87
src/transformers/modeling_xlm.py:87
src/transformers/modeling_xlm.py:87
src/transformers/modeling_xlm.py:87
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:87: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    assert mask.size() == (bs, slen)

src/transformers/modeling_xlm.py:450
src/transformers/modeling_xlm.py:450
src/transformers/modeling_xlm.py:450
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:450: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in t
he future. This means that the trace might not generalize to other inputs!
    assert lengths.size(0) == bs

src/transformers/modeling_xlm.py:451
src/transformers/modeling_xlm.py:451
src/transformers/modeling_xlm.py:451
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:451: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in th
e future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen

src/transformers/modeling_xlm.py:451
src/transformers/modeling_xlm.py:451
src/transformers/modeling_xlm.py:451
  /home/msrauser/project/transformers/src/transformers/modeling_xlm.py:451: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in t
he future. This means that the trace might not generalize to other inputs!
    assert lengths.max().item() <= slen
src/transformers/modeling_openai.py:166
src/transformers/modeling_openai.py:166
src/transformers/modeling_openai.py:166
  /home/msrauser/project/transformers/src/transformers/modeling_openai.py:166: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in
the future. This means that the trace might not generalize to other inputs!
    w = w / math.sqrt(v.size(-1))

src/transformers/modeling_openai.py:169
src/transformers/modeling_openai.py:169
src/transformers/modeling_openai.py:169
  /home/msrauser/project/transformers/src/transformers/modeling_openai.py:169: TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in
the future. This means that the trace might not generalize to other inputs!
    b = self.bias[:, :, : w.size(-2), : w.size(-1)]

src/transformers/modeling_mobilebert.py:512
src/transformers/modeling_mobilebert.py:512
src/transformers/modeling_mobilebert.py:512
  /home/msrauser/project/transformers/src/transformers/modeling_mobilebert.py:512: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables t
hat would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
    + s

-- Docs: https://docs.pytest.org/en/latest/warnings.html
======================================================================================================================== short test summary info =========================================================================================================================
FAILED tests/test_modeling_electra.py::ElectraModelTest::test_multigpu_data_parallel_forward - RuntimeError: tensor.ndimension() == static_cast<int64_t>(expected_size.size()) INTERNAL ASSERT FAILED at /pytorch/torch/csrc/cuda/comm.cpp:225, please report a bug to ...
FAILED tests/test_modeling_gpt2.py::GPT2ModelTest::test_multigpu_data_parallel_forward - RuntimeError: Gather got an input of invalid size: got [2, 2, 4, 7, 8], but expected [2, 4, 4, 7, 8] (gather at /pytorch/torch/csrc/cuda/comm.cpp:231)
FAILED tests/test_modeling_bart.py::BartHeadTests::test_lm_forward - timeout_decorator.timeout_decorator.TimeoutError: 'Timed Out'
FAILED tests/test_modeling_ctrl.py::CTRLModelTest::test_multigpu_data_parallel_forward - RuntimeError: CUDA error: out of memory
FAILED tests/test_modeling_xlnet.py::XLNetModelTest::test_multigpu_data_parallel_forward - RuntimeError: CUDA error: out of memory
FAILED tests/test_modeling_openai.py::OpenAIGPTModelTest::test_multigpu_data_parallel_forward - RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 15.78 GiB total capacity; 5.33 MiB already allocated; 17.19 MiB free; 6.00 MiB reserved in total ...
FAILED tests/test_modeling_mobilebert.py::MobileBertModelTest::test_multigpu_data_parallel_forward - RuntimeError: CUDA error: out of memory
================================================================================================= 7 failed, 1066 passed, 562 skipped, 1396 warnings in 333.27s (0:05:33) =================================================================================================
Makefile:19: recipe for target 'test' failed
make: *** [test] Error 1

```
