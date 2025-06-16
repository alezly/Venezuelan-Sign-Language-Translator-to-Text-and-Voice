import { Component, ViewChild } from '@angular/core';
import { SignCameraComponent } from '../sign-camera/sign-camera.component'

@Component({
  selector: 'app-sign-capture',
  imports: [SignCameraComponent],
  templateUrl: './sign-capture.component.html',
  styleUrl: './sign-capture.component.scss'
})
export class SignCaptureComponent {
  @ViewChild(SignCameraComponent) userCameraComponent!: SignCameraComponent;

  selected: string = 'camera';

  select(name: string) {
    this.selected = name;
  }


  activarCamara() {
    this.userCameraComponent.startCamera();
  }

  desactivarCamara() {
    this.userCameraComponent.stopCamera();
  }
}
