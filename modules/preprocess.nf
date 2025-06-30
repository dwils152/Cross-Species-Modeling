#!/usr/bin/env nextflow
nextflow.enable.dsl=2

workflow PREPROCESS_GENOME {
    take:
        genome_and_organism
    main:
        tokenize_genome(genome_and_organism)
    emit:
        tokenize_genome.out.map {
            it -> tuple(it[0], it[2])
        }
}

process tokenize_genome {
    label "MedMem"
    publishDir "${params.results}/${organism}", mode: 'copy' 
    input:
        tuple   path(genome),
                val(organism)
    output:
        tuple   path("*.chunks.fa"),
                path("*.chunks.bed"),
                val(organism)
    script:
        """
        python ${params.scripts}/tokenize_genome_fast.py \
            -i ${genome}
        """
}
