# 705.603 - Creating AI-Enabled Systems

## Tarun Nadipalli (tnadipa1@jh.edu)

Hey! This is my repository for the 705.603 Creating AI-Enabled Systems class taught by Professor John Hebeler in the JHU Masters in AI.

### Course Description

_“Achieving the full capability of AI requires a system perspective to effectively leverage algorithms, data, and computing power. Creating AI-enabled systems includes thoughtful consideration of an operational decomposition for AI solutions, engineering data for algorithm development, and deployment strategies. To realize the impact of AI technologies requires a systems perspective that goes beyond the algorithms. The objective of this course is to bring a system perspective to creating AI-enabled systems. The course will explore the full-lifecycle of creating AI-enabled systems starting with problem decomposition and addressing data, design, diagnostic, and deployment phases.”_

---
### Module Structure

This class contains weekly homework assignments that are broken up into modules. Each module usually consists of the following: 

- Jupyter Notebook
- Python script
- Dockerfile
- requirements.txt
- dataset

Within each module there is a more descriptive README that will outline instructions for running the scripts within a docker environment and links for the docker images.  

---
### Setup

To setup the coding environment in which all of this code was written, please follow the instructions here.

**Installations**

1. Install Docker [here](https://docs.docker.com/get-docker/).
2. Set up a Dockerhub account [here](https://hub.docker.com/). This is where you will be uploading your Docker images.
3. Setup a GitHub account [here](https://github.com/signup) to push your code to.

**Local Setup**
1. Download the class image from [here](https://hub.docker.com/r/jhebeler/classtank/tags?page=1&ordering=last_updated). We will pull this image and use it to start up our working JupyterLab environment.
    
    ```bash
    docker pull jhebeler/classtank:705.603.jupyterlab
    ```
    Check that you see the image using:
    ```bash
    docker image ls
    ```
2. Start the Container
  
    Let's first run through the `docker run` command and it's arguments to understand what is necessary to start the container.

    ```
    docker run –restart=unless-stopped -it -p <local port>:<container port> -v <host directory>:<container directory> <image>
    ```
    - -restart: controls when the container is restarted. This option restarts the container all the time unless stopped by the docker stop command. This includes restarting it on a reboot.
    - -p: maps a host port to a container port
    - -v: maps a host directory to a container directory

    Run in command/terminal window. Note: the host directory MUST exist prior to running the command. For
    example, the command below maps /workspace on the host file system to /rapids/notebooks/workspace within
    the container file system. /workspace MUST exist. It need not be /workspace. It can be any directory you want
    to use on your host system to share files such as code with the container. /workspace is just an example.

    Windows:
    ``` 
    docker run --restart=unless-stopped -it -p 8888:8888 -p 8787:8787 -p5000:5000 -v c:/workspace:/rapids/notebooks/workspace jhebeler/classtank:705.603.jupyterlab 
    ```

    Linux/MAC:
    ``` 
    docker run --restart=unless-stopped -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 5000:5000 -v /workspace:/rapids/notebooks/workspace jhebeler/classtank:705.603.jupyterlab 
    ```
    
    *Note: Sometimes the command above may throw an error about the pathing. If that is the case, paste the full path to the workspace directory in the <host directory> portion of the command.

 3. Start JupyterLab within the container
 
    Good job! You are now inside the container. This is due to the -it flag which places you directly inside the container with a new prompt.
    Now you need to start JupyterLab in the correct directory. Navigate there (the shared directory!) with the command below.
    ```
    cd /rapids/notebooks/workspace
    ```
    Now start up JupyterLab.
    ```
    jupyter lab --no-browser --ip=0.0.0.0 --allow-root
    ```
    
    *Note: You can also start JupyterLab from outside the container using this command.
    ```
    docker exec -it -w /rapids/notebooks/workspace <Container ID> jupyter lab --ip=0.0.0.0 --allow-root
    ```
    To get the container ID, use this command (You can also use -a to get inactive containers):
    ```
    docker ps
    ```
    
    Now you should have a working JupyterLab at port http://localhost:8888 (JupyterLab default workspace)!

    For the first time, you will need to copy the token provided in the container window to log into the workspace.
    You can set up separate work spaces via http://localhost:8888/lab/workspaces/<NAME>. This allows you to
    save context such as open files etc. for different workspaces.
    Leave this container terminal window up. If you reboot the system, the container also restarts but you have to
    restart jupyter lab in either of the ways above.
    
    You're good to go! Happy coding!
    
    _Special thanks to Professor Hebeler for providing all of the instructions above!_
    
    
    
    
    
    
    
    
    
    
