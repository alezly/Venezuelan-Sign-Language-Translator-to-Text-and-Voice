import { Component } from '@angular/core';

@Component({
  selector: 'app-sign-capture',
  imports: [],
  templateUrl: './sign-capture.component.html',
  styleUrl: './sign-capture.component.scss'
})
export class SignCaptureComponent {
  selected: string = 'camera';

  select(name: string) {
    this.selected = name;
  }
}
