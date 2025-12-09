import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { BotUpdate } from './bot.update';
import { TelegrafModule } from 'nestjs-telegraf';

@Module({
  imports: [TelegrafModule.forRoot({
    token: "8562528084:AAE_A5GGM75oKKsEIOcnghZf080QSjJ05gM"
  })],
  controllers: [AppController],
  providers: [AppService, BotUpdate],
})
export class AppModule {}
