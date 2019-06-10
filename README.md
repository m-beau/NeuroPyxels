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
  - /npix:
    - gl.py:
    - spk_t.py:
    - spk_wvf.py:
