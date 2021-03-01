import React, { Component } from "react";
import ImageUpload from "./components/ImageUpload";
import "./App.css";

class App extends Component {
  state = {
    prediction: null
  };
  onFormSubmit = prediction => {
    this.setState({ prediction: prediction });
  };
  render() {
    return (
      <div>
        <ImageUpload onFormSubmit={this.onFormSubmit} />
      </div>
    );
  }
}

export default App;
