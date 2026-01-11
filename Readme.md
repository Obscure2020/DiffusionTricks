# Diffusion Tricks Project

Developed in April/May 2025 by Joey Sodergren and Akash Prasad as an exploratory final project for Advanced Artificial Intelligence at Wright State University.

## Initial Setup
To set up the Python virtual environment and download the SDXL model weights for this project, perform the following sequence of steps.

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
The LaTeX source of our project writeup is present in the `latex_source` directory. Before attempting to compile the document into a readable PDF, we recommend that you ensure your system meets these requirements:

- An up-to-date installation of [TeX Live](https://www.tug.org/texlive/) should be present and ready to use.
- Times New Roman and Cascadia Code should be installed as system fonts.

If those conditions have been satisfied, then you can run the following command in the `latex_source` directory to compile the document:
```
latexmk -interaction=nonstopmode -file-line-error -pdf -xelatex "Diffusion Tricks.tex" && latexmk -c "Diffusion Tricks.tex"
```