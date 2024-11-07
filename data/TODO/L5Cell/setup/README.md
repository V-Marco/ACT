# Environment Setup

## Installing Miniconda

[[Link to their website]](https://docs.conda.io/projects/miniconda/en/latest/)

In your home directory, execute the following:
```bash
mkdir ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
The last command will ask you to restart the shell. After re-logging, you should see the `(base)` environment being active. It will now be active every time you connect to the server.

## Creating a new environment

Create a new environment called `CHOOSE_NAME` (choose any name you want) with
```bash
conda create -n CHOOSE_NAME python=3.10
```
When it asks you if you want to proceed, type `y`.

## Activating the environment

You can activate this environment with
```bash
conda activate CHOOSE_NAME
```

You should do it each time you connect to the server.

## Installing packages

Install required packages with
```bash
cd setup
pip install -r requirements.txt
```

## Cloning the cell_inference library

To clone the latest version of `cell_inference`, you can do
```bash
cd setup
bash clone_cell_inference.sh
```

# Runtime

## Screen

1.  Create a screen session named `sim` (you can choose any name)
    ```bash
    screen -S sim
    ```
    This will open a new, separate terminal window.
2. By default, the screen starts with the `base` conda environment. Activate a different environment if needed. Start a simulation.
3. To exit screen, press `Ctrl + A + D`. This will send you back to the main terminal window.
4.  To return to a screen, execute
    ```bash
    screen -ls
    ```
    in the main terminal window. It will display a list of active screens. Then execute
    ```bash
    screen -r [screen name]
    ```
    If you have only one screen opened, you can just do
    ```bash
    screen -r
    ```
    in the main terminal window.
5.  Once the simulation is finished, execute
    ```bash
    exit
    ```
    in the screen window to close it.
