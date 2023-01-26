function get_single_prediction(user_record) {
    $.ajax({
        url: "stream",
        type: "POST",
        headers: {
        'Content-Type': 'application/json'
        },
        data: JSON.stringify({ data: user_record }),
        success: function (data) {
            const ans = '<div class="answer">'
                + '<div class="answer_text"><p style="white-space: pre-line;">' + data + '</p><i></i>'
                + '</div></div>';

            $('.speak_box').append(ans);
            fit_screen();
        },
        error: function (XMLHttpRequest) {
            alert("Server error：" + XMLHttpRequest.status);
        }
    });
}

function get_multiple_predictions(user_records) {
    $.ajax({
        url: "batch",
        type: "POST",
        headers: {
        'Content-Type': 'application/json'
        },
        data: JSON.stringify({ data: user_records }),
        success: function (data) {
            const ans = '<div class="answer">'
                + '<div class="answer_text"><p style="white-space: pre-line;">' + data + '</p><i></i>'
                + '</div></div>';

            $('.speak_box').append(ans);
            fit_screen();
        },
        error: function (XMLHttpRequest) {
            alert("Server error：" + XMLHttpRequest.status);
        }
    });
}

function key_up() {
    var text = $('.chat_box input').val();
    if (text == '') {
        $('.write_list').remove();
    } else {
        const str = '<div class="write_list">' + text + '</div>';
        $('.footer').append(str);
        $('.write_list').css('bottom', $('.footer').outerHeight());
    }
}

function fit_screen() {
    $('.speak_box, .speak_window').animate({scrollTop: $('.speak_box').height()}, 500);
}

function auto_width() {
    $('.question_text').css('max-width', $('.question').width() - 60);
}

function send_payload() {
    $('.write_list').remove();
    const userInput = $('.chat_box input').val();

    if (userInput == '') {
        alert('please enter your message');
        $('.chat_box input').focus();
        $('body').css('background-image', 'url(resources/images/bg.png)');
    } else {
        const str = '<div class="question">'
            + '<div class="question_text clear"><p style="white-space: pre-line;">' + userInput + '</p><i></i>'
            + '</div></div>';

        $('.speak_box').append(str);
        $('.chat_box input').val('');
        $('.chat_box input').focus();

        fit_screen();
        auto_width();

        const isMultiPredict = userInput.includes(',');
        if (isMultiPredict){
            get_multiple_predictions(userInput)
        }else{
            get_single_prediction(userInput)
        }

    }
}
