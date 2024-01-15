import cv2
import numpy as np
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    Create a histogram with k clusters
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    """
    Plot the colors in the histogram
    """
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the central rectangle dimensions
        h, w, _ = frame.shape
        centerX, centerY = w // 2, h // 2
        rectWidth, rectHeight = 100, 100  # Adjust size of the rectangle
        startX, startY = centerX - rectWidth // 2, centerY - rectHeight // 2
        endX, endY = centerX + rectWidth // 2, centerY + rectHeight // 2

        # Extract the central rectangle
        rect = frame[startY:endY, startX:endX]
        rect = rect.reshape((rect.shape[0] * rect.shape[1], 3))

        # Apply KMeans clustering
        clt = KMeans(n_clusters=1)  # We want the most dominant color
        clt.fit(rect)

        hist = find_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)

        # Display the dominant color bar
        cv2.imshow("Dominant Color", bar)

        # Display the frame with a rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
