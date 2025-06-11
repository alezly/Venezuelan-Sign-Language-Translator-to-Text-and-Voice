import { Component } from '@angular/core';

@Component({
  selector: 'app-text-output',
  imports: [],
  templateUrl: './text-output.component.html',
  styleUrl: './text-output.component.scss'
})
export class TextOutputComponent {
  selected: string = 'Text output';

  select(name: string) {
    this.selected = name;
  }
}
