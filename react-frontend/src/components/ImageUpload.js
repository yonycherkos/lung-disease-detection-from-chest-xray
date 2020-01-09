import React from "react";
import axios from "axios";
import "./ImageUpload.css";

class ImageUpload extends React.Component {
  state = {
    filename: "Choose Image",
    filepath: `${process.env.PUBLIC_URL}/images/Line-Drawing-chest.jpg`,
    base64Image: null,
    prediction: [],
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

  onInputChange = event => {
    if (event.target.files[0] != null) {
      let filename = event.target.files[0].name;
      let filepath = URL.createObjectURL(event.target.files[0]);

      let reader = new FileReader();
      reader.readAsDataURL(event.target.files[0]);
      reader.onload = () => {
        let dataURL = reader.result;
        let base64Image = dataURL.replace("data:image/png;base64,", "");
        this.setState({
          filename,
          filepath,
          base64Image,
          predicted_class_name: "",
          predicted_class_value: ""
        });
      };
    } else {
      console.log("image is not selected");
    }
  };

  onFormSubmit = event => {
    event.preventDefault();
    const url = "http://127.0.0.1:5000/predict";
    let message = {
      image: this.state.base64Image
    };
    axios
      .post(url, message)
      .then(response => {
        this.props.onFormSubmit(response.data.prediction);
        let prediction = response.data.prediction[0];
        let predicted_class_name = this.state.class_names[
          prediction.indexOf(Math.max(...prediction))
        ];
        let predicted_class_value = (Math.max(...prediction) * 100).toFixed(2);
        this.setState({
          prediction: prediction,
          predicted_class_name: predicted_class_name + ": ",
          predicted_class_value: predicted_class_value + "%"
        });
        console.log(response.data.prediction[0]);
      })
      .catch(error => console.log(error));
  };

  render() {
    return (
      <div className="container mt-5 main">
        <h1 className="text-center">General X-Ray Imaging</h1>
        <div className="col-sm-12 col-lg-4 mr-auto ml-auto border p-4 shadow sm mb-5 bg-white rounded">
          <form
            action="http://127.0.0.1:5000/predict"
            method="post"
            encType="multipart/form-data"
            onSubmit={this.onFormSubmit}
          >
            <div className="form-group">
              <label>
                <strong>Upload Images</strong>
              </label>
              <div className="custom-file">
                <input
                  type="file"
                  name="imageFile"
                  className="custom-file-input"
                  id="customFile"
                  onChange={this.onInputChange}
                />
                <label className="custom-file-label" htmlFor="customFile">
                  {this.state.filename}
                </label>
              </div>
            </div>
            <div className="z-depth-1-half mb-4">
              <img
                src={this.state.filepath}
                className="img-fluid"
                alt={this.state.filename}
              ></img>
            </div>
            <div className="form-group">
              <button
                type="submit"
                name="upload"
                value="upload"
                id="upload"
                className="btn btn-block btn-primary"
              >
                <i className="fa fa-fw fa-upload"></i> Upload
              </button>
            </div>
          </form>
          <div>
            <h3>Prediction Results</h3>
            <p>
              {this.state.predicted_class_name +
                this.state.predicted_class_value}
            </p>
          </div>
        </div>
      </div>
    );
  }
}

export default ImageUpload;
