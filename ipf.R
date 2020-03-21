options(java.parameters = "-Xmx8000m")

library(NoiseFiltersR)

datos <- as.data.frame(read.csv("/tmp/data_for_r.txt"))
datos_limpios<-IPF(datos, p=0.5, s=4)$cleanData
labels <- datos_limpios[,length(datos_limpios)]
datos_limpios<-datos_limpios[,-length(datos_limpios)]

write.table(datos_limpios, file="/tmp/data_for_python.txt", col.names = FALSE, row.names = FALSE)
write.table(labels, file="/tmp/labels_for_python.txt", col.names = FALSE, row.names = FALSE)
