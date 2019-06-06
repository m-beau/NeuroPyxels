# routines

***************************************************************************************************
Formal Definition of a "routine" -> a python function with the following properties:
- embedded in a script called routines_xxx.py -> accessed by calling routines_xxx.routine()
- computes a commonly used Variable, computed from a dataset. The Variable is a numpy array.
- when called, looks for the Variable in dir/routinesMemory where dir is the directory of 
  the dataset. If not found, the routine saves the variable after having computed it.
- by default, routines do not return the Variable. Rather they upload it to 
  the global namesSpace of the python session calling the routine. In this case, 
  the routine prints out the name of the computed Variable.
***************************************************************************************************
