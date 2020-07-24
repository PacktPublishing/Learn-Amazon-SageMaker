library('plumber')

source('train_function.R')

serve <- function() {
    app <- plumb('serve_function.R')
    app$run(host='0.0.0.0', port=8080)}

args <- commandArgs()
if (any(grepl('train', args))) {
    train()
}
if (any(grepl('serve', args))) {
    serve()
}

