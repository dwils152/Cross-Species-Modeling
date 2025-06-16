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





