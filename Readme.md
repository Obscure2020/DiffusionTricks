# Diffusion Tricks Project

Developed in April/May 2025 by Joey Sodergren and Akash Prasad an exploratory final project for Advanced Artificial Intelligence at Wright State University.

## Initial Setup
To set up the python virtual environment and download the SDXL model weights for this project, perform the following sequence of steps.

1. Open a terminal into the directory containing this repo.
1. Grant execute permissions to the initial setup script.
    ```sh
    chmod 755 initSetup.sh
    ```
1. Run the setup script itself.
    ```sh
    ./initSetup.sh
    ```

**Warning:** Depending on the download speed of your internet connection, the initial setup script may take a very long time to complete its work.

The setup script will tell you when it is complete by printing this message to the console:
```
Project environment setup complete.
```

## Running
To execute the the project itself, simply perform the following in the directory containing this repo:
```sh
source venv/bin/activate
python3 main.py
deactivate
```

## Documentation
The LaTeX source of our project writeup is present in the `latex_source` directory. **Note:** the document may not compile unless the following conditions are satisfied:
- The document is compiled through XeLaTeX for extended font support
- Times New Roman and Cascadia Code are installed as system fonts.