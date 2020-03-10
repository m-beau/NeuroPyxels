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

```
RTNinstall(){
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
}```

Or directly from the command line:

# Push changes to remote github repo
cd path/to/gitRepo
git add.
git commit -m "message"
git push

# Reinstall python module from local git repo
conda activate gabi
pip uninstall rtn
python setup.py develop #(could also install from remote repo by doing: pip ginstall git+http://m-beau/routines@Gabi)

# Test that fix works...
python
>>> import rtn
>>> rtn blahblahblah # NO ERROR YIHAAAA

# Pull request to Maxime's master branch using the online github green button...
