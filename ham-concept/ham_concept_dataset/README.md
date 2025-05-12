Reader study was conducted in 3 phases: no ai (phase 1), ai support (phase 2), xai support (phase 3). The data generated in each phase is available in metadata_phase1.csv, metadata_phase2.csv, and metadata_phase3.csv.

There are 113 unique participants in phase 1. Additional participants were added in phase 2, resulting in 116 unique clinicians in phases 2 and 3.

**Important:**  The 3rd and 13th image in each group are identical. Be careful when performing table joins as the duplicate image_ids can affect them. In metadata_phase1.csv, the AI predictions for the 13th image in each group are null. Please take that into account when performing analysis. In metadata_phase2 and metadata_phase3, the AI predictions for the repeating images are not omitted.

  

_participant_: Each clinician was assigned a participant Id represented by the participant column.

_group_: Each clinician was randomly assigned to a group. Each group was assigned mutually exclusive sets of images.

_mask_: An internal identifier used for the images. Can be ignored.

_benign_malignant_: ground truth diagnosis.

_prediction_: Diagnosis chosen by the clinician. 1 represents melanoma, 0 represents nevus, 0.5 represents a nevus diagnosis but the clinician chose to excise.

_confidence_: Confidence value entered by the clinician.

_trust_: Trust value entered by the clinician.

_AI_prediction_: Diagnosis predicted by the AI. 1 represents melanoma and 0 represents nevus.

_language_: Language chosen by the clinician.