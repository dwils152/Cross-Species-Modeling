# Cross-Species-Modeling

## Install Nextflow:
Nextflow is a workflow manager that will, submit jobs to the HPC

1. Nextflow requires java which is already on the cluster so you can load it with:
   ```bash
   module load openjdk/22
   ```
2. You will need to load this every time you use Nextflow so I would include this in your .bashrc file.
   ```bash
   echo "module load openjdk/22" >> ~/.bashrc";
   source ~/.bashrc
   ```
   
3. Use this command to download Nextflow
   ```bash
   curl -s https://get.nextflow.io | bash
   ```
   
4. Move the Nextflow binary to your `bin` folder. I would recommend moving it to `/users/$USER/.local/bin` because it should already be in your PATH.
   ```bash
   mv nextflow /users/$USER/.local/bin
   ```
   
5. Test your installation by running `nextflow run hello`

## Install the conda environment:
To run the code you will need to install all of its dependencies. They are all grouped together in a conda environment.

1. Download the code from the repository onto the cluster and move into the directory
   ```bash
   git clone https://github.com/dwils152/Cross-Species-Modeling.git;
   cd Cross-Species-Modeling
   ```

2. Load Anaconda
   ```bash
   module load anaconda3
   ```
   
3. Build the environment from the `environment.yml` file
  ```bash
     conda env create -f environment.yml -n cross_species
  ```

4. Activate the environment
   ```
   conda activate cross_species
   ```

Now you should have installed all the resources you need to generate predictions.




