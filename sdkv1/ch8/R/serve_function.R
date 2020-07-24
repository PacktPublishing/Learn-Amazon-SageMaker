#' @get /ping
function() {
    return('')}

#' @param req The http request sent
#' @post /invocations
function(req) {    
    model <- readRDS('/opt/ml/model/model.rds')
    conn <- textConnection(gsub('\\\\n', '\n', req$postBody))
    data <- read.csv(conn)
    close(conn)
    print(data)
    medv <- predict(model, data)
    return(medv)
}