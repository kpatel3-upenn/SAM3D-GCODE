import cv2
import numpy as np
import os
import argparse
import json

# Function to redraw the entire image with points, lines, coordinates, and instructions
def redraw_image():
    global img, base_img, pos_points, neg_points, current_phase

    img = base_img.copy()  # Reset the image to the original without drawings


    # Draw all positive polylines
    for polyline in pos_points:
        for i, point in enumerate(polyline):
            cv2.circle(img, point, 2, (0, 255, 0), -1)  # Green for positive points
            # cv2.putText(img, str(point), (point[0] + 5, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            if i > 0:
                cv2.line(img, polyline[i - 1], point, (0, 255, 0), thickness=1)

    # Draw all negative polylines
    for polyline in neg_points:
        for i, point in enumerate(polyline):
            cv2.circle(img, point, 2, (0, 0, 255), -1)  # Red for negative points
            # cv2.putText(img, str(point), (point[0] + 5, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            if i > 0:
                cv2.line(img, polyline[i - 1], point, (0, 0, 255), thickness=1)

    # Draw instructions
    instructions = "Left-click to draw. 'A' to switch. 'W' for new pos line." "\n" "'S' for new neg line. 'D' to delete. 'Q' to quit."
    phase_instruction = "Drawing " + ("Positive (Green)" if current_phase == "positive" else "Negative (Red)") + " Polylines"

    # Split the instructions into lines
    instruction_lines = instructions.split('\n')

    # Starting Y position for the first line
    y0 = 15

    # Loop through each line and draw it
    for i, line in enumerate(instruction_lines):
        # Adjust Y position for each line (15 pixels between lines as an example)
        y = y0 + i * 15
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw the phase instruction below the last instruction line
    cv2.putText(img, phase_instruction, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow('image', img)

# Mouse callback function for drawing points and lines
def click_event(event, x, y, flags, param):
    global pos_points, neg_points, current_phase

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        if current_phase == "positive":
            pos_points[-1].append(point)  # Add to the last list of positive polylines
            pos_points_tosave[-1].append((y, x, 0))

        else:
            neg_points[-1].append(point)  # Add to the last list of negative polylines
            neg_points_tosave[-1].append((y, x, 0))
        redraw_image()  # Redraw the image with the new point

# Function to start a new polyline
def start_new_polyline(polyline_type):
    global pos_points, neg_points, current_phase
    if polyline_type == "positive":
        pos_points.append([])  # Start a new list for a new positive polyline
        pos_points_tosave.append([])
        current_phase = 'positive'
    else:
        neg_points.append([])  # Start a new list for a new negative polyline
        neg_points_tosave.append([])
        current_phase = 'negative'
    # print(f"Started a new {'positive' if polyline_type == 'positive' else 'negative'} polyline.")
    redraw_image()  # Redraw the image to update the instructions and visible points

# Function to start a new polyline
def delete_point():
    global pos_points, neg_points, current_phase
    if current_phase == "positive":
        if len(pos_points[-1]) == 0:
            if len(pos_points) == 1:
                print('no points left to delete')
                return
            pos_points.pop()
            pos_points_tosave.pop()
        else:
            pos_points[-1].pop()
            pos_points_tosave[-1].pop()
    else:
        if len(neg_points[-1]) == 0:
            if len(neg_points) == 1:
                print('no points left to delete')
                return
            neg_points.pop()
            neg_points_tosave.pop()
        else:
            neg_points[-1].pop()
            neg_points_tosave[-1].pop()
    redraw_image()  # Redraw the image to update the instructions and visible points

# Function to switch between positive and negative points collection
def switch_phase():
    global current_phase
    current_phase = "negative" if current_phase == "positive" else "positive"
    if current_phase == "positive":
        pos_points.append([])
        pos_points_tosave.append([])
    else:
        neg_points.append([])
        neg_points_tosave.append([])
    # print(f"Switched to {'negative' if current_phase == 'negative' else 'positive'} points collection.")
    redraw_image()  # Redraw the image to update the instructions and visible points


def main(folder='tempdir'):
    listofdicts = []
    filepaths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])
    # Copy of the original image to use as a base for redrawing
    for filepath in filepaths:

        img = cv2.imread(filepath)
        global base_img, pos_points, pos_points_tosave, neg_points, neg_points_tosave, current_phase
        base_img = img.copy()
        # Initialize global variables
        pos_points = [[]]  # List of lists to hold positive polylines
        pos_points_tosave = [[]]
        neg_points = [[]]  # List of lists to hold negative polylines
        neg_points_tosave = [[]]
        current_phase = "positive"  # Start with collecting positive points

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)  # Initial display
        cv2.resizeWindow("image", 800, 800)
        cv2.setMouseCallback('image', click_event)
        redraw_image()  # Initial drawing of instructions

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):  # Switch phases
                switch_phase()
            elif key == ord('w'):  # Start a new positive polyline
                start_new_polyline("positive")
            elif key == ord('s'):  # Start a new negative polyline
                start_new_polyline("negative")
            elif key == ord('d'):  # delete a point or line (if last point)
                delete_point()
            elif key == ord('q'):  # Quit
                break
        cv2.destroyAllWindows()
        pos_points_tosave = [el for el in pos_points_tosave if el != []]
        neg_points_tosave = [el for el in neg_points_tosave if el != []]
        # if len(pos_points_tosave)==0: pos_points_tosave.append([])
        # if len(neg_points_tosave)==0: neg_points_tosave.append([])
        dictionary = {"img": filepath, "pos_polylines": pos_points_tosave, "neg_polylines": neg_points_tosave}
        listofdicts.append(dictionary)
    filename = folder + '/prompts.json'

    # Write the list of dictionaries to the file in JSON format
    with open(filename, 'w') as f:
        json.dump(listofdicts, f, indent=4)

    # print(f"Data has been saved to {filename}")
    return listofdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the slice directory")
    args = parser.parse_args()
    main(args.path)