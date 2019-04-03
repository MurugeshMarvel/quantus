import glob
import json
import os
import shutil
import operator
import sys
import argparse

MINOVERLAP = 0.5

def get_map_score():
  ignore = []
  quite = None
  specific_iou_flagged = False


  show_animation = False

  def error(msg):
    print(msg)
    sys.exit(0)

  def is_float_between_0_and_1(value):
    try:
      val = float(value)
      if val > 0.0 and val < 1.0:
        return True
      else:
        return False
    except ValueError:
      return False

  def voc_ap(rec, prec):
 
    rec.insert(0, 0.0) 
    rec.append(1.0) 
    mrec = rec[:]
    prec.insert(0, 0.0) 
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
   
    for i in range(len(mpre)-2, -1, -1):
      mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
      if mrec[i] != mrec[i-1]:
        i_list.append(i) 
    ap = 0.0
    for i in i_list:
      ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

  def file_lines_to_list(path):
    with open(path) as f:
      content = f.readlines()
    content = [x.strip() for x in content]
    return content

  def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        color,
        lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

  def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])
  tmp_files_path = "tmp_files"
  if not os.path.exists(tmp_files_path):
    os.makedirs(tmp_files_path)
  results_files_path = "results"
  if os.path.exists(results_files_path): 
    shutil.rmtree(results_files_path)

  os.makedirs(results_files_path)

  if show_animation:
    os.makedirs(results_files_path + "/images")
    os.makedirs(results_files_path + "/images/single_predictions")

  ground_truth_files_list = glob.glob('val_dir/ground-truth/*.txt')
  if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
  ground_truth_files_list.sort()
  gt_counter_per_class = {}

  for txt_file in ground_truth_files_list:
    #print(txt_file)
    file_id = txt_file.split(".txt",1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    if not os.path.exists('val_dir/predicted/' + file_id + ".txt"):
      error_msg = "Error. File not found: val_dir/predicted/" +  file_id + ".txt\n"
      error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
      error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    bounding_boxes = []
    is_difficult = False
    lines_list = lines_list[1:]
    for line in lines_list:
      try:
        if "difficult" in line:
            class_name, left, top, right, bottom, _difficult = line.split()
            is_difficult = True
        else:
            class_name, left, top, right, bottom = line.split()
      except ValueError:
        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
        error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
        error_msg += " Received: " + line
        error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
        error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
        error(error_msg)
      if class_name in ignore:
        continue
      bbox = left + " " + top + " " + right + " " +bottom
      if is_difficult:
          bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
          is_difficult = False
      else:
          bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
          if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
          else:
            gt_counter_per_class[class_name] = 1
    with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
      json.dump(bounding_boxes, outfile)

  gt_classes = list(gt_counter_per_class.keys())
  gt_classes = sorted(gt_classes)
  n_classes = len(gt_classes)
  #print(gt_classes)
  #print(gt_counter_per_class)

  predicted_files_list = glob.glob('val_dir/predicted/*.txt')
  predicted_files_list.sort()

  for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in predicted_files_list:
      #print(txt_file)
      file_id = txt_file.split(".txt",1)[0]
      file_id = os.path.basename(os.path.normpath(file_id))
      if class_index == 0:
        if not os.path.exists('val_dir/ground-truth/' + file_id + ".txt"):
          error_msg = "Error. File not found: val_dir/ground-truth/" +  file_id + ".txt\n"
          error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
          error(error_msg)
      lines = file_lines_to_list(txt_file)
      lines = lines[1:]
      for line in lines:
        try:
          tmp_class_name, confidence, left, top, right, bottom = line.split()
        except ValueError:
          error_msg = "Error: File " + txt_file + " in the wrong format.\n"
          error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
          error_msg += " Received: " + line
          error(error_msg)
        if tmp_class_name == class_name:
          #print("match")
          bbox = left + " " + top + " " + right + " " +bottom
          bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
          #print(bounding_boxes)

    bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
    with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
      json.dump(bounding_boxes, outfile)

  sum_AP = 0.0
  ap_dictionary = {}
  with open(results_files_path + "/results.txt", 'w') as results_file:
    results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
      count_true_positives[class_name] = 0
      predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
      predictions_data = json.load(open(predictions_file))
      nd = len(predictions_data)
      tp = [0] * nd 
      fp = [0] * nd
      for idx, prediction in enumerate(predictions_data):
        file_id = prediction["file_id"]
        if show_animation:
          ground_truth_img = glob.glob1(img_path, file_id + ".*")
          if len(ground_truth_img) == 0:
            error("Error. Image not found with id: " + file_id)
          elif len(ground_truth_img) > 1:
            error("Error. Multiple image with id: " + file_id)
          else:
            img = cv2.imread(img_path + "/" + ground_truth_img[0])
            img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
            if os.path.isfile(img_cumulative_path):
              img_cumulative = cv2.imread(img_cumulative_path)
            else:
              img_cumulative = img.copy()
            bottom_border = 60
            BLACK = [0, 0, 0]
            img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        ovmax = -1
        gt_match = -1
        bb = [ float(x) for x in prediction["bbox"].split() ]
        for obj in ground_truth_data:
          if obj["class_name"] == class_name:
            bbgt = [ float(x) for x in obj["bbox"].split() ]
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
              ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                      + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
              ov = iw * ih / ua
              if ov > ovmax:
                ovmax = ov
                gt_match = obj

        if show_animation:
          status = "NO MATCH FOUND!" 
        min_overlap = MINOVERLAP
        if specific_iou_flagged:
          if class_name in specific_iou_classes:
            index = specific_iou_classes.index(class_name)
            min_overlap = float(iou_list[index])
        if ovmax >= min_overlap:
          if "difficult" not in gt_match:
              if not bool(gt_match["used"]):
                tp[idx] = 1
                gt_match["used"] = True
                count_true_positives[class_name] += 1
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
                if show_animation:
                  status = "MATCH!"
              else:
                fp[idx] = 1
                if show_animation:
                  status = "REPEATED MATCH!"
        else:
          fp[idx] = 1
          if ovmax > 0:
            status = "INSUFFICIENT OVERLAP"
        if show_animation:
          height, widht = img.shape[:2]
          white = (255,255,255)
          light_blue = (255,200,100)
          green = (0,255,0)
          light_red = (30,30,255)
          margin = 10
          v_pos = int(height - margin - (bottom_border / 2.0))
          text = "Image: " + ground_truth_img[0] + " "
          img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
          text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
          img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
          if ovmax != -1:
            color = light_red
            if status == "INSUFFICIENT OVERLAP":
              text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
            else:
              text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
              color = green
            img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
          v_pos += int(bottom_border / 2.0)
          rank_pos = str(idx+1) 
          text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
          img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
          color = light_red
          if status == "MATCH!":
            color = green
          text = "Result: " + status + " "
          img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

          font = cv2.FONT_HERSHEY_SIMPLEX
          if ovmax > 0: 
            bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
            cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
            cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
            cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
          bb = [int(i) for i in bb]
          cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
          cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
          cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
          cv2.imshow("Animation", img)
          cv2.waitKey(20) 
          output_img_path = results_files_path + "/images/single_predictions/" + class_name + "_prediction" + str(idx) + ".jpg"
          cv2.imwrite(output_img_path, img)
          cv2.imwrite(img_cumulative_path, img_cumulative)

      cumsum = 0
      for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
      cumsum = 0
      for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
      #print(tp)
      rec = tp[:]
      for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
      #print(rec)
      prec = tp[:]
      for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
      #print(prec)

      ap, mrec, mprec = voc_ap(rec, prec)
      sum_AP += ap
      text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " 

      rounded_prec = [ '%.2f' % elem for elem in prec ]
      rounded_rec = [ '%.2f' % elem for elem in rec ]
      results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
 
      ap_dictionary[class_name] = ap


    if show_animation:
      cv2.destroyAllWindows()

    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    results_file.write(text + "\n")
    #print(text)

  shutil.rmtree(tmp_files_path)

  pred_counter_per_class = {}
  for txt_file in predicted_files_list:
    lines_list = file_lines_to_list(txt_file)
    lines_list =lines_list[1:]
    for line in lines_list:
      class_name = line.split()[0]
      if class_name in ignore:
        continue
      if class_name in pred_counter_per_class:
        pred_counter_per_class[class_name] += 1
      else:
        pred_counter_per_class[class_name] = 1
  #print(pred_counter_per_class)
  pred_classes = list(pred_counter_per_class.keys())

  with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
      results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

  for class_name in pred_classes:
    if class_name not in gt_classes:
      count_true_positives[class_name] = 0
  #print(count_true_positives)

  with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of predicted objects per class\n")
    for class_name in sorted(pred_classes):
      n_pred = pred_counter_per_class[class_name]
      text = class_name + ": " + str(n_pred)
      text += " (tp:" + str(count_true_positives[class_name]) + ""
      text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
      results_file.write(text)
  return (mAP*100)