# routines

***************************************************************************************************
Formal Definition of a "routine" -> a python function with the following properties:
- embedded in a script called routines_xxx.py -> accessed by calling routines_xxx.routine()
- computes a commonly used Variable, computed from a dataset. The Variable is a numpy array.
- when called, looks for the Variable in dir/routinesMemory where dir is the directory of 
  the dataset. If not found, the routine saves the variable after having computed it.
***************************************************************************************************

Python version has to be >3.7 ot there will be imports issues!

Commands to properly push modifications to your github repo, packed as a bash function (put tht in your ~/.bashrc):

# TO BE RUN IN A GIVEN CONDA ENVIRONMENT

`RTNinstall(){
    pip install git+https://www.github.com/m-beau/routines.git
}

RTNpush(){
    if [ $# -eq 0 ] # Forces to supply a commit message
    then
        echo "No argument supplied - exiting now. Provide a commit message!"
    else
        pip uninstall rtn
        cd ~/Dropbox/routines
        git diff
        git checkout master
        git add .
        git commit -m "$1"
        git push origin master
        RTNinstall
    fi
}`
