#train <- read.csv("/Users/gregorymatthews/Dropbox/nfl-big-data-bowl-2020/train.csv")
#save.image("/Users/gregorymatthews/Dropbox/bigDataBowl_2019/BigDataBowl_Data_2019.RData")
load("/Users/gregorymatthews/Dropbox/bigDataBowl_2019/BigDataBowl_Data_2019.RData")

#Mike Lopez Stuff
train <- train %>% 
  mutate(ToLeft = PlayDirection == "left", 
         IsBallCarrier = NflId == NflIdRusher)

train$VisitorTeamAbbr <- as.character(train$VisitorTeamAbbr)
train$HomeTeamAbbr <- as.character(train$HomeTeamAbbr)

train$VisitorTeamAbbr[train$VisitorTeamAbbr == "ARI"] <- "ARZ"
train$HomeTeamAbbr[train$HomeTeamAbbr == "ARI"] <- "ARZ"

train$VisitorTeamAbbr[train$VisitorTeamAbbr == "BAL"] <- "BLT"
train$HomeTeamAbbr[train$HomeTeamAbbr == "BAL"] <- "BLT"

train$VisitorTeamAbbr[train$VisitorTeamAbbr == "CLE"] <- "CLV"
train$HomeTeamAbbr[train$HomeTeamAbbr == "CLE"] <- "CLV"

train$VisitorTeamAbbr[train$VisitorTeamAbbr == "HOU"] <- "HST"
train$HomeTeamAbbr[train$HomeTeamAbbr == "HOU"] <- "HST"



train <- train %>% 
  mutate(YardsFromOwnGoal = ifelse(as.character(FieldPosition) == PossessionTeam, 
                                   YardLine, 50 + (50-YardLine)), 
         YardsFromOwnGoal = ifelse(YardLine == 50, 50, YardsFromOwnGoal))


#Create Offense and Defense Variables
train$side[as.character(train$PossessionTeam) == as.character(train$HomeTeamAbbr) & train$Team == "home"] <- "offense"
train$side[as.character(train$PossessionTeam) == as.character(train$HomeTeamAbbr) & train$Team == "away"] <- "defense"

train$side[as.character(train$PossessionTeam) != as.character(train$HomeTeamAbbr) & train$Team == "home"] <- "defense"
train$side[as.character(train$PossessionTeam) != as.character(train$HomeTeamAbbr) & train$Team == "away"] <- "offense"



#remove plays with missing Dir and Orientation
rem <- union(train$PlayId[is.na(train$Dir)], train$PlayId[is.na(train$Orientation)])
train <- train[!train$PlayId%in%rem,]

train$PlayId <- as.character(train$PlayId)
pids <- as.character(unique(train$PlayId))

##first of all make 0 degrees up and down the field
train$Orientation_new <- train$Orientation + 90
train$Dir_new <- train$Dir + 90

#First make sure every play is going left to right to standardize
#Im only fixing the X and Y coordinates for now
train$X[train$PlayDirection == "left"] <- 120 - train$X[train$PlayDirection == "left"] - 10 #why minus 10? So 0 is the 0 yard line
train$Y[train$PlayDirection == "left"] <- 53.3 - train$Y[train$PlayDirection == "left"] 

train$Orientation_new[train$PlayDirection == "left"] <- train$Orientation_new[train$PlayDirection == "left"] + 180
train$Dir_new[train$PlayDirection == "left"] <- train$Dir_new[train$PlayDirection == "left"] + 180

train$Orientation_new[train$Orientation_new > 360 ] <- train$Orientation_new[train$Orientation_new > 360 ] - 360
train$Dir_new[train$Dir_new > 360 ] <- train$Dir_new[train$Dir_new > 360 ] - 360


#Make everytihng radians
train$Orientation_new <- train$Orientation_new/180*pi
train$Dir_new <- train$Dir_new/180*pi

train$PlayDirection[train$PlayDirection == "left"] <- "right"


#sub <- subset(train, PlayId == pids[1])
#library(ggplot2)
#ggplot(aes(x = X, y= Y, col = side, shape = (radial_speed >0)) , data = sub) + geom_point() + xlim(0,100) + ylim(0,160/3) + geom_point(aes(x = X, y= Y) , data = sub[1,], col = "red")


ingrid <- function(pid, rrr_off , rrr_def, theets ){
#pull out one play to test on 
#sub <- subset(train, PlayId == pids[1])
sub <- subset(train, PlayId == pid)
print(pid)
#Setting up the data
#Creating the grid
#Pull out the running back 
rb <- subset(sub, sub$NflId == sub$NflIdRusher)
center <- c(rb$X, rb$Y)

sub$dist_from_RB <- sqrt((sub$X - center[1])^2 + (sub$Y - center[2])^2)
sub$ang_from_RB <- acos((sub$X-center[1])/sub$dist_from_RB)
#sub$ang_from_RB[sub$Y < center[2]] <- -sub$ang_from_RB[sub$Y < center[2]]

#sub$dist_from_RB this is like r
#sub$ang_from_RB this is like theta. 
sub$radial_speed <- sub$S*cos(sub$ang_from_RB - sub$Dir_new)
sub$tangential_speed <- sub$S*sin(sub$ang_from_RB - sub$Dir_new)

sub_def <- subset(sub, side == "defense")
sub_off <- subset(sub, side == "offense")

def_dat <- unlist(c(sub_def[order(sub_def$dist_from_RB),c("Orientation_new","Dir_new","dist_from_RB","ang_from_RB","radial_speed","tangential_speed")]))
off_dat <- unlist(c(sub_off[order(sub_off$dist_from_RB),c("Orientation_new","Dir_new","dist_from_RB","ang_from_RB","radial_speed","tangential_speed")][-1,]))

grid_list_def <- list()

for (r in 2:length(rrr_off)){
#r <- 1

grid_list_def[[rrr_def[r]]] <- rep(NA, length(theets)-1)
for (i in 1:(length(theets)-1)){
  grid_list_def[[rrr_def[r]]][i] <- (sum(sub_def$dist_from_RB > (rrr_def[r-1]) & sub_def$dist_from_RB <= rrr_def[r] & sub_def$ang_from_RB > theets[i] & sub_def$ang_from_RB <= theets[i+1]))
}

}


grid_list_off <- list()

for (r in 2:length(rrr_off)){
  #r <- 1
  grid_list_off[[rrr_off[r]]] <- rep(NA, length(theets)-1)
  for (i in 1:(length(theets)-1)){
    grid_list_off[[rrr_off[r]]][i] <- (sum(sub_off$dist_from_RB > (rrr_off[r-1]) & sub_off$dist_from_RB <= rrr_off[r] & sub_off$ang_from_RB > theets[i] & sub_off$ang_from_RB <= theets[i+1]))
  }
  
}

  
return(c(unlist(grid_list_def),unlist(grid_list_off)))
  
}

start <- Sys.time()
grids <- list()
rrr_off <- c(0,1:15)
rrr_def <- c(0,1:15,20,25,30,35,40,45,50)
theets <- seq(-pi,pi,pi/8)

rrr_off <- c(0,seq(1,15,3))
rrr_def <- c(0,seq(1,15,3))
theets <- seq(-pi,pi,pi/4)
test <- lapply(as.list(pids), ingrid, rrr_def = rrr_def ,rrr_off = rrr_off, theets = theets)
names(test) <- pids
end <- Sys.time()
end-start

grid_dat <- as.data.frame(do.call(rbind,test))
colnames(grid_dat) <- c(paste0("def_r_",rep(rrr_def[-1],each = length(theets[-1])),"_theta_",rep(1:length(theets[-1]),length(rrr_def[-1]))),paste0("off_r_",rep(rrr_off[-1],each = length(theets[-1])),"_theta_",rep(1:length(theets[-1]),length(rrr_off[-1]))))
  
grid_dat$PlayId <- pids




#Now pull out only the rushing player data
train_rush <- subset(train, NflId == NflIdRusher)


dat <- merge(grid_dat, train_rush, by.x = "PlayId", by.y = "PlayId")

write.csv(dat, file = "/Users/gregorymatthews/Dropbox/bigDataBowl_2019/grid.csv")

write.csv(dat, file = "/Users/gregorymatthews/Dropbox/bigDataBowl_2019/grid.csv")

dat <- read.csv("/Users/gregorymatthews/Dropbox/bigDataBowl_2019/grid.csv")


library(glmnet)
#Now build a model
dat$Y <- (dat$Yards <= 0) + 0
form <- formula(paste0("Y~",paste0(c(paste0("def_r_",rep(rrr_def[-1],each = length(theets[-1])),"_theta_",rep(1:length(theets[-1]),length(rrr_def[-1]))),paste0("off_r_",rep(rrr_off[-1],each = length(theets[-1])),"_theta_",rep(1:length(theets[-1]),length(rrr_off[-1])))),collapse = "+")))
test <- glm(form, family = "binomial", data = dat)









ingrid2 <- function(pid){
  #pull out one play to test on 
  #sub <- subset(train, PlayId == pids[1])
  sub <- subset(train, PlayId == pid)
  print(pid)
  #Setting up the data
  #Creating the grid
  #Pull out the running back 
  rb <- subset(sub, sub$NflId == sub$NflIdRusher)
  center <- c(rb$X, rb$Y)
  
  sub$dist_from_RB <- sqrt((sub$X - center[1])^2 + (sub$Y - center[2])^2)
  sub$ang_from_RB <- acos((sub$X-center[1])/sub$dist_from_RB)
  #sub$ang_from_RB[sub$Y < center[2]] <- -sub$ang_from_RB[sub$Y < center[2]]
  
  #sub$dist_from_RB this is like r
  #sub$ang_from_RB this is like theta. 
  sub$radial_speed <- sub$S*cos(sub$ang_from_RB - sub$Dir_new)
  sub$tangential_speed <- sub$S*sin(sub$ang_from_RB - sub$Dir_new)
  
  sub_def <- subset(sub, side == "defense")
  sub_off <- subset(sub, side == "offense")
  
  def_dat <- unlist(c(sub_def[order(sub_def$dist_from_RB),c("Orientation_new","Dir_new","dist_from_RB","ang_from_RB","radial_speed","tangential_speed")]))
  off_dat <- unlist(c(sub_off[order(sub_off$dist_from_RB),c("Orientation_new","Dir_new","dist_from_RB","ang_from_RB","radial_speed","tangential_speed")][-1,]))

  return(c(def_dat,off_dat))
  
}


start <- Sys.time()
test <- lapply(as.list(pids), ingrid2)
names(test) <- pids
end <- Sys.time()
end-start

grid_dat <- as.data.frame(do.call(rbind,test))
grid_dat$PlayId <- pids
names(grid_dat)[1:66] <- paste0("def_",names(grid_dat)[1:66])
names(grid_dat)[67:126] <- paste0("def_",names(grid_dat)[67:126])

#Now pull out only the rushing player data
train_rush <- subset(train, NflId == NflIdRusher)

dat <- merge(grid_dat, train_rush, by.x = "PlayId", by.y = "PlayId")
write.csv(dat, file = "/Users/gregorymatthews/Dropbox/bigDataBowl_2019/peter_data.csv")

