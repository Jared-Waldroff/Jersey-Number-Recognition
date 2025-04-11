from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# Set the local directory where you want to save the dataset
mySNdl = SNdl(LocalDirectory="./data/SoccerNet")
# Download train, test, and challenge splits for the jersey number task
mySNdl.downloadDataTask(task="jersey-2023", split=["test"])
