import { Component } from '@angular/core';
import { NgClass } from '@angular/common';
@Component({
  selector: 'app-toggle-switch',
  imports: [NgClass],
  templateUrl: './toggle-switch.component.html',
  styleUrl: './toggle-switch.component.scss',
})
export class ToggleSwitchComponent {
  isToggled = false;

  // Este método se llama cuando el usuario hace clic en el interruptor
  toggle(): void {
    this.isToggled = !this.isToggled;
    console.log('Toggle state:', this.isToggled); // Opcional: para ver el estado en la consola
  }
}
