import { Component, ViewChild, Output, EventEmitter } from '@angular/core';
import { SignCameraComponent } from '../sign-camera/sign-camera.component'

@Component({
  selector: 'app-sign-capture',
  imports: [SignCameraComponent],
  templateUrl: './sign-capture.component.html',
  styleUrl: './sign-capture.component.scss'
})
export class SignCaptureComponent {
  @ViewChild(SignCameraComponent) userCameraComponent!: SignCameraComponent;
  @Output() translationChanged = new EventEmitter<string>();
  selected: string = 'camera';

  onTranslationChanged(value: string) {
    this.translationChanged.emit(value);
  }

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
