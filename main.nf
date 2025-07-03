#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { PREPROCESS_GENOME    } from "./modules/preprocess.nf"
include { CROSS_SPECIES_PRED   } from "./modules/cross_species_pred.nf"

workflow  {

        deepspeed_config = Channel.value("${projectDir}/ds_config_zero2.json")
        model            = Channel.value("${projectDir}/model/human_model")
        lora_adapter     = Channel.value("${projectDir}/model/human_LoRA")

        samples_ch = Channel
            .fromPath(params.sample_sheet)
            .splitCsv(
                header: true,
                sep: ","
            )
            .map { row -> tuple(row.genome, row.organism) }

        PREPROCESS_GENOME( samples_ch )

        CROSS_SPECIES_PRED( 
            PREPROCESS_GENOME.out,
            deepspeed_config,
            model,
            lora_adapter
        )


}