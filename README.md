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

```
