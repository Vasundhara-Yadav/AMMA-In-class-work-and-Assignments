
normal_income=rnorm(100,mean=250000,sd=75000)
normal_income
mean<-mean(normal_income)
sd<-sd(normal_income)

gender=c(rep("F",100))
i=sample(1:100,100,replace=FALSE)
for(q in 1:100){
  if(gender[i[q]] == "F" &&  q <= 40)
  {}
  else
    gender[i[q]] = c("M")
}
gender
Custs <-data.frame(ID=1:100,INCOME= normal_income, GENDER = gender)
View(Custs)
