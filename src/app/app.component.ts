import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SignCaptureComponent } from "./sign-capture/sign-capture.component";
import { TextOutputComponent } from "./text-output/text-output.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, SignCaptureComponent, TextOutputComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'LSV-to-text-and-speech';
}
