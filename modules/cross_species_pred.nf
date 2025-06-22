#!/users/dwils152/bin nextflow
nextflow.enable.dsl=2

workflow CROSS_SPECIES_PRED {
    take:
        fasta_and_organism
        deepspeed_config
        model
        lora
    main:

        fasta = fasta_and_organism.map{ it[0] }
        organism = fasta_and_organism.map{ it[1] }

        predict(
            fasta,
            organism,
            deepspeed_config,
            model,
            lora
        )
}

process predict {
    label "GPU"
    publishDir "${params.results}/${organism}/Predictions", mode: 'copy' 
    input:
        path(fasta)
        val(organism)
        path(deepspeed_config)
        path(model)
        path(lora)
    output:
        tuple   path("predictions.csv"),
                val(organism)
    script:
        """
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        deepspeed --num_gpus 4 ${params.scripts}/deepspeed-cross_spec.py \
            --fasta  ${fasta} \
            --model_checkpoint ${model} \
            --lora_dir ${lora} \
            --deepspeed ${deepspeed_config}
        """
}