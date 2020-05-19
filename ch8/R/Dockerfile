FROM r-base:latest

COPY train.R /opt/ml/train.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/train.R", "--no-save"]