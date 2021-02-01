# AmongUsLobbyData

Will analyze a 1080p streamer video on demand (VoD) by taking a snapshot every 2 minutes to extract a
meeting call in the game Among Us, and use OCR engine Tesseract to pull the names of those
in the game to get an indication of who is playing with who.

To allow for inaccuracies in OCR capture, similar names will be grouped by Levenshtein Distance Ratio 
and number of "hits" - the higher the number the more often that name showed up.


More details in [my medium article](https://kleenip.medium.com/my-misadventures-with-opencv-tesseract-and-among-us-75458ed211ad)



