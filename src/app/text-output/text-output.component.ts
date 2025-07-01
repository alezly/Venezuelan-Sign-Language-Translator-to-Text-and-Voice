import { Component, Input  } from '@angular/core';
import { ToggleSwitchComponent } from '../toggle-switch/toggle-switch.component';

@Component({
  selector: 'app-text-output',
  imports: [ToggleSwitchComponent],
  templateUrl: './text-output.component.html',
  styleUrl: './text-output.component.scss'
})
export class TextOutputComponent {
  @Input() translation: string = '';
  selected: string = 'Text output';

  select(name: string) {
    this.selected = name;
  }
}
