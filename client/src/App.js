import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import Webcam from "react-webcam";
import { Button } from 'reactstrap';
import { Card, CardImg, CardText, CardBody,
    CardTitle, CardSubtitle } from 'reactstrap';
import 'bootstrap/dist/css/bootstrap.min.css';


class App extends Component {

  constructor()
  {
    super();
    this.state = {
      video: null
    }
  }


  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <p>Find out how hot our Artificial Intelligence thinks you are!</p>
        </div>
        <div>
            <Card>
                <CardBody className={"camArea"}>
                    <Webcam
                        screenshotFormat="image/jpeg"
                    />
                    <br/>


                    <Button className={"upload-button"}>Upload Photo</Button>
                </CardBody>
            </Card>
        </div>
        <div className="App-footer">
            <a href={"https://electricbrain.io/"}>
                <img src={"/EB_Brain_White.svg"}  className={"eb-logo"}/>
            </a>
        </div>
      </div>
    );
  }
}

export default App;
