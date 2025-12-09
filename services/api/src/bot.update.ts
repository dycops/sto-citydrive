import { Update, Start, Help, On, Hears, Ctx } from 'nestjs-telegraf';
import { Context } from 'telegraf';

@Update()
export class BotUpdate {
  @Start()
  async onStart(@Ctx() ctx: Context) {
    await ctx.reply('Привет! Я NestJS Telegraf бот.');
  }

  @Help()
  async onHelp(@Ctx() ctx: Context) {
    await ctx.reply('Отправь мне фото или напиши команду.');
  }

  @On('photo')
  async onPhoto(@Ctx() ctx: Context) {
    await ctx.reply('Фото получено, обрабатываю...');
  }

  @Hears('ping')
  async onPing(@Ctx() ctx: Context) {
    await ctx.reply('pong');
  }
}