library(psych)
dir = "C:\\Users\\Willi\\Documents\\NYU\\Research\\NLU\\crowdsourcing\\IRT\\nlu-test-sets\\data\\"

answers <- read.csv(paste(dir,"combined_irt_all_coded.csv",sep=""),header=TRUE,sep=",")

R<-tetrachoric(answers)
plot(eigen(R$rho,symmetric=TRUE,only.values=TRUE)$values)