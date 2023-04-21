word_aprox <- function(x,y,p=c(0.3)){
  #---------------------------------------------------*
  #PASO 1: COINCIDENCIAS EN PALABRAS
  #---------------------------------------------------*
  #TOKENIZANDO
  Y <- sapply(y, tok,USE.NAMES = F)
  X <- sapply(x, tok,USE.NAMES = F)
  #---------------------------------------------------*
  #IDENTIFICA PALABRA MÁS FRECUENTE
  mf_tq <- data.frame(data=unlist(tokenize_words(X))) %>% group_by(data) %>% tally() %>% filter(n>1)
  mf_tq <- paste0(mf_tq$data,collapse = "|")
  #---------------------------------------------------*
  # ELIMINANDO PALABRA MAS FRECUENTE: QUECHUA, MATSIGENKA, YARU
  Y <- str_trim(ifelse(sapply(str_split(Y, "\\s+"),length)>1,gsub(pattern = mf_tq,"",Y),Y)) #tokenize_words(unique(C_TEMP$D_LENGUA),lowercase = F)
  X <- str_trim(ifelse(sapply(str_split(X, "\\s+"),length)>1,gsub(mf_tq,"",X),X)) #tokenize_words(lo_cv$deslengua,lowercase = F)
  #---------------------------------------------------*
  #CREANDO MATRIZ DE COINCIDENCIAS
  Z <- matrix(nrow = length(Y),ncol = length(X),data = NA)
  JW <- matrix(nrow = length(Y),ncol = length(X),data = NA)
  temp_Y <- tokenize_words(Y)
  temp_X <- tokenize_words(X)
  #---------------------------------------------------*
  for (i in 1:length(X)) {
    for (j in 1:length(Y)) {
      #j=8;i=4
      l <- length(intersect(temp_Y[[j]], temp_X[[i]]))#; print(l)
      k <- 1-stringdist(Y[j], X[i], method = "jw",p=0.1)
      #print(c(Y[j],X[i],l,j)
      Z[j,i] <- l
      JW[j,i] <- k
    }
  }
  #---------------------------------------------------*
  #IDENTIFICANDO INDICES QUE SUPEREN UN % DE COINCIDENCIAS
  indices <- which(Z/rowSums(Z) >= p, arr.ind = TRUE)
  indices <- as.data.frame(indices)
  res_1 <- unique(y)[indices$row]
  res_2 <- unique(x)[indices$col]
  res <- cbind.data.frame(res_1,res_2)
  res_fin_1 <- res %>% filter(res_1!=res_2)
  res_fin_1
  #---------------------------------------------------*
  rowmax <- apply(JW, MARGIN = 1, max)
  #rowmax <- ifelse(rowmax<0.72,1,rowmax)
  indices <- which(JW >=rowmax , arr.ind = TRUE)
  indices <- as.data.frame(indices)
  #---------------------------------------------------*
  res_1 <- unique(y)[indices$row]
  res_2 <- unique(x)[indices$col]
  res <- cbind.data.frame(res_1,res_2)
  res_fin_2<- res %>% filter(res_1!=res_2) %>% anti_join(res_fin_1,by=c("res_1"))
  res_fin <- rbind.data.frame(res_fin_1,res_fin_2) 
  return(res_fin)
}
