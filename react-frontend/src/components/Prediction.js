import React from "react";
import "./Prediction.css";

class Prediction extends React.Component {
  state = {
    class_names: [
      "Atelectasis",
      "Cardiomegaly",
      "Consolidation",
      "Edema",
      "Effusion",
      "Emphysema",
      "Fibrosis",
      "Hernia",
      "Infiltration",
      "Mass",
      "Nodule",
      "Pleural_Thickening",
      "Pneumonia",
      "Pneumothorax"
    ],
    predicted_class_name: "",
    predicted_class_value: ""
  };

  onPrediction = () => {
    let predicted_class_name = this.state.class_names[
      Math.max(this.props.prediction)
    ];
    let predicted_class_value = Math.max(this.props.prediction) * 100;
    this.setState({ predicted_class_name, predicted_class_value });
  };

  render() {
    return (
      <div>
        <h3>Prediction Results</h3>
        <p>
          {this.state.predicted_class_name +
            ": " +
            this.state.predicted_class_value +
            "%"}
        </p>
      </div>
    );
  }
}

export default Prediction;
