# routines

***************************************************************************************************
Formal Definition of a "routine" -> a python function with the following properties:
- embedded in a script called routines_xxx.py -> accessed by calling routines_xxx.routine()
- computes a commonly used Variable, computed from a dataset. The Variable is a numpy array.
- when called, looks for the Variable in dir/routinesMemory where dir is the directory of 
  the dataset. If not found, the routine saves the variable after having computed it.
***************************************************************************************************

List of routines:

- /routines_utils.py: lists general purpose python utilitaries
- /Neuropixels
  - /routines_spikes.py:
    - Dataset: Neuropixels dataset -> dp is phy directory (kilosort output)
    - Neuropixels datasets are based on units. Here, only units spikes will be adressed (arrays of time stamps).
      Hence the argument ul.
    - Neuropixels datsets can be accessed either through XtraDataManager
      or saved exports from phy with the snippets :export_samples, :export_times and :export_ids.
      Hence the argument src.
