

### 2026-10-28T13:38:51.738224+00:00 — Short segments and CUDA ownership correction prepared

The user authorized fixing launch/storage issues and shorter runs. New precision
continuation16 keeps Pro20k->50k/LR0.3 and DBMol7.5k->10k/LR0.0025, using2500/1250
update segments with exact saved boundary/RNG handoffs and all original scientific
checks. New same-job exclusive-L40S MPS request plus real two-client FP32/FP64 GPU
smoke precedes model loading; previous source15 CUDA/storage failures remainfailed.
Measured fullpilot runtime determines eachsegment walltime including bothreference
sessions/cleanup. Newexplicitrollingcapacity counts currentroots once, funds only
active/admitted/nextbounded outputs and4GiB/file reserves; laterunchangedtargets must
re-admit, no implicit lifetime reservation or deletion. All numerical validation
remainsallocated; localAST/metadata/byte implementation is not a passedfix.
At13:31:56UTC onlyindependentDeFoG29839884RUN, source16remote pathsabsent; three
externalA100jobsPEND and oldMosaicreport29796558PEND. Existing savedruns untouched,
monitorpaused. FreshplainBol20k record reveals persistentopposite-sign atomAD/FD,
soqueue/MPS changes cannotadmitit. FreshProchem nestedlog identifies staleFP64
integercharge/asinh downcast (production exactreplaypassed); correctedreference09
requiresfreshchem-specificqualification. No newjob/test/model/cleanup executedyet.
Owner: experiments/validation/endpoint-qualification/runs/20261028-precision-continuation-16/.
