# Tips for Windows User

## Run make test on Windows terminal

- [Install the windows subsystem for linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) 
- After enabling WSL2, install ubuntu from Microsoft store.
- [Enable the Hyper-V](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v)
- Open the ubuntu terminal and install python3
  - Use `python3 –version` to verify python3 is installed.
- Open pycharm and follow the steps below:
  1.	Go to File->Settings->Tools->Terminal
  2.	In application setting select the shell path for the ubuntu.
  3.	Default path will be: C:\Users\name\AppData\Local\Microsoft\WindowsApps\ubuntu2004.exe
  To reach to this path you must select "Show hidden files and directories".
  4.	Now you have got an ubuntu terminal
  - In ubuntu terminal select the directory to be SMIDA 
  - Install all the required packages like pandas using command `sudo apt-get install python3-pandas`. Install other packages in same way.
  - Install make package using `sudo apt-get install make`
- Now you can run make test in terminal