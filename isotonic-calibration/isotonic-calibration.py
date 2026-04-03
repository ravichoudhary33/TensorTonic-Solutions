from sklearn.isotonic import IsotonicRegression

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    # Write code here
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(cal_probs, cal_labels)
    return ir.predict(new_probs).tolist() 